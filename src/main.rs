use itertools::Itertools;
use rand_distr::{Distribution, Normal, Triangular, Uniform};
use rayon::{
    prelude::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelBridge, ParallelIterator,
    },
    slice::ParallelSliceMut,
};
use realfft::RealFftPlanner;
use rubato::{FftFixedInOut, Resampler};
use std::{
    fs::{read_dir, File},
    ops::Range,
    path::Path,
    sync::Mutex,
};

// Sampling rate of the audio while the genetic algorithm runs.
// Lower values will be much faster but will yield lower final audio fidelity.
const HZ: u32 = 20000;
// How many seconds per chunk each sample audio is cut into when intially loaded.
const SAMPLE_CHUNK_SECS: f32 = 1.0;
// How large each chunk of target audio we work on are. Must be greater than SAMPLE_CHUNK_SECS.
const TARGET_CHUNK_SECS: f32 = 5.0;
// How many generations will run. More generations means reaching a better fitting solution.
const GENERATIONS: u32 = 50;
// This is how many offspring each sample will create (not taking into account two offspring per one split transformation).
const OFFSPRINGS_PER_SAMPLE: usize = 3;
// The max number of samples that can ever survive from one generation. Take this as the overpopulation threshold.
const GENERATION_CAPACITY: usize = 500;
// Number of the best samples are going to be added to the output audio per generation.
const OUTPUT_CAPACITY: usize = 10;
// How many seconds of tolerance are going to be accepted when comparing nearby delays of other output samples.
// If the delay of a sample is too close to a delay that is already outputted in the same generation, it will be skipped.
const NEARBY_DELAY_TOLERANCE: f32 = 0.3;
// How many seconds of silence is tolerated before it is removed from the sample audio.
const SILENCE_TOLERANCE: f32 = 0.5;
// The top percentile of population that survive to transform/crossover.
const SURVIVAL_RATIO: f32 = 0.5;
// Generation index modulo for resetting the fitness seeding.
// As in, if this is e.g., 10, then every 10 generations the fitness will be recalculated from scratch (these generations are significantly slower to compute).
const SEED_RESET_MOD: u32 = 25;
// Percentage of surviving population that is chosen (at random) to be crossed over (signals added together).
const CROSSOVER_RATIO: f32 = 0.1;

// The number of elites selected per generation is calculated as
//      population_count * ELITISM_RATIO * 2^(ELITISM_COEFF * (generation / total_generations) - ELITISM_COEFF)
// This is so that it can "ramp up" as we approach the end of our generation.

// Thew highest percentage of elitism possible.
// Essentially, on the very last generation, this is how many percent of the population will be taken as elites.
// It's recommended that if you're running a low number of generations, that this be equal or very close to 0.
const ELITISM_RATIO: f32 = 0.;
// How much the elitism is scaled through the generations.
// The higher this number is, the more delayed elitism will be.
// Must be >= 0. To disable "dynamic elitism", set this value to 0 and the elitism ratio will remain constant through every generation.
const ELITISM_COEFF: f32 = 10.0;

fn freq_pitch(cents: i32) -> u32 {
    (2.0f32.powf(cents as f32 / 1200.0) * HZ as f32) as _
}

fn secs_to_frames(secs: f32) -> usize {
    (HZ as f32 * secs) as _
}

fn frames_to_secs(frames: usize) -> f32 {
    frames as f32 / HZ as f32
}

#[derive(Clone)]
struct Sample {
    pub id: u64,
    pub signal: [Vec<f32>; 2], // stereo
    pub state: Option<(u32, f32)>,
    pub seed: Option<u32>,
    pub gain: f32,
}

impl Sample {
    fn new(len: usize) -> Self {
        Sample {
            id: 0,
            signal: [vec![0.; len], vec![0.; len]],
            state: None,
            seed: None,
            gain: 1.,
        }
    }

    fn from_wav(file: impl AsRef<Path>) -> Self {
        let mut file = File::open(file).expect("read file");
        let (header, data) = wav::read(&mut file).expect("decode wav");
        let interleaved = match data {
            wav::BitDepth::Sixteen(samples) => samples
                .into_iter()
                .map(|sample| sample as f32 / 0x8000 as f32)
                .collect::<Vec<_>>(),
            _ => unimplemented!(),
        };
        let (l, r) = interleaved.chunks(2).map(|x| (x[0], x[1])).unzip();
        Sample {
            id: 0,
            signal: [l, r],
            state: None,
            seed: None,
            gain: 1.,
        }
        .resample(header.sampling_rate, HZ)
    }

    fn save_to_wav(&self, file: impl AsRef<Path>) {
        let mut file = File::create(file).expect("write file");
        wav::write(
            wav::Header::new(wav::WAV_FORMAT_PCM, 2, HZ, 16),
            &wav::BitDepth::Sixteen(
                self.signal[0]
                    .iter()
                    .interleave(self.signal[1].iter())
                    .map(|x| (x.clamp(-1., 1.) * 0x8000 as f32) as i16)
                    .collect(),
            ),
            &mut file,
        )
        .expect("encode wav");
    }

    fn len(&self) -> usize {
        self.signal[0].len()
    }

    fn chunked(&self, frames: usize) -> Vec<Self> {
        self.signal[0]
            .chunks(frames)
            .zip(self.signal[1].chunks(frames))
            .map(|(l, r)| Sample {
                id: 0,
                signal: [l.to_vec(), r.to_vec()],
                state: None,
                seed: None,
                gain: self.gain,
            })
            .collect()
    }

    fn resample(&self, f1: u32, f2: u32) -> Self {
        const CHUNK_SIZE: usize = 2048;

        if f1 == f2 {
            return self.clone();
        }

        let mut resampler =
            FftFixedInOut::new(f1 as _, f2 as _, CHUNK_SIZE, 2).expect("init resampler");
        let mut left_out = vec![0.; resampler.output_frames_max()];
        let mut right_out = vec![0.; resampler.output_frames_max()];

        let mut remaining = self.signal[0].len();
        let mut next_frames = resampler.input_frames_next();
        while remaining > next_frames {
            let mut out = resampler
                .process(
                    &[
                        &self.signal[0][self.signal[0].len() - remaining..],
                        &self.signal[1][self.signal[1].len() - remaining..],
                    ],
                    None,
                )
                .expect("resample");
            left_out.append(&mut out[0]);
            right_out.append(&mut out[1]);
            remaining -= next_frames;
            next_frames = resampler.input_frames_next();
        }

        if remaining > 0 {
            let mut out = resampler
                .process_partial(
                    Some(&[
                        &self.signal[0][self.signal[0].len() - remaining..],
                        &self.signal[1][self.signal[1].len() - remaining..],
                    ]),
                    None,
                )
                .expect("resample");
            left_out.append(&mut out[0]);
            right_out.append(&mut out[1]);
        }

        Sample {
            id: self.id,
            signal: [left_out, right_out],
            state: None,
            seed: self.state.map(|(x, _)| x),
            gain: self.gain,
        }
    }

    fn reverse(&self) -> Self {
        let mut out = self.signal.clone();
        out.reverse();
        Sample {
            id: self.id,
            signal: out,
            state: None,
            seed: self.state.map(|(x, _)| x),
            gain: self.gain,
        }
    }

    fn split(&self, at: usize) -> [Self; 2] {
        let ((a_left, b_left), (a_right, b_right)) =
            (self.signal[0].split_at(at), self.signal[1].split_at(at));
        [
            Sample {
                id: self.id,
                signal: [a_left.to_vec(), a_right.to_vec()],
                state: None,
                seed: self.state.map(|(x, _)| x),
                gain: self.gain,
            },
            Sample {
                id: self.id,
                signal: [b_left.to_vec(), b_right.to_vec()],
                state: None,
                seed: self.state.map(|(x, _)| x),
                gain: self.gain,
            },
        ]
    }

    fn pitch(&self, cents: i32) -> Self {
        self.resample(HZ, freq_pitch(cents))
    }

    fn amplitude(&self, ratio: f32) -> Self {
        let ratio = (self.gain * ratio).clamp(0.6, 1.5) / self.gain;

        if (self.gain * ratio - self.gain).abs() < f32::EPSILON {
            return Sample {
                id: self.id,
                signal: self.signal.clone(),
                state: None,
                seed: self.state.map(|(x, _)| x),
                gain: self.gain,
            };
        }

        let left = self.signal[0].par_iter().map(|x| x * ratio).collect();
        let right = self.signal[1].par_iter().map(|x| x * ratio).collect();
        Sample {
            id: self.id,
            signal: [left, right],
            state: None,
            seed: self.state.map(|(x, _)| x),
            gain: self.gain * ratio,
        }
    }

    fn transforms(&self, n: usize, target: &Sample, out: &mut Vec<Sample>, max_len: &mut usize) {
        let type_dist = Uniform::new(0, 5);
        let split_dist = Uniform::new(self.len() / 4, 3 * self.len() / 4);
        let pitch_dist = Uniform::new(-200, 200);
        let amplitude_dist = Normal::new(1.0f32, 0.2).expect("normal distribution");
        let false_seed_dist = Uniform::new(0, (target.len() - self.len()).max(1));
        let mut rng = rand::thread_rng();
        (0..n).for_each(|_| match type_dist.sample(&mut rng) {
            0 => {
                out.push(self.reverse());
                *max_len = (*max_len).max(self.len());
            }
            1 => {
                out.extend_from_slice(&self.split(split_dist.sample(&mut rng)));
                *max_len = (*max_len).max(self.len() / 2);
            }
            2 => {
                out.push(
                    self.pitch(pitch_dist.sample(&mut rng))
                        .truncate(target.len()),
                );
                *max_len = (*max_len).max(self.len());
            }
            3 => {
                out.push(self.amplitude(amplitude_dist.sample(&mut rng).clamp(0.6, 1.5)));
                *max_len = (*max_len).max(self.len());
            }
            4 => out.push(Sample {
                id: self.id,
                signal: self.signal.clone(),
                state: None,
                seed: Some(false_seed_dist.sample(&mut rng) as _),
                gain: self.gain,
            }),
            _ => unreachable!(),
        });
    }

    fn add(&self, rhs: &Sample) -> Self {
        if self.len() >= rhs.len() {
            let left = self.signal[0]
                .par_iter()
                .enumerate()
                .map(|(i, x)| (x + self.signal[0][i]) / 2.)
                .collect();
            let right = self.signal[1]
                .par_iter()
                .enumerate()
                .map(|(i, x)| (x + self.signal[1][i]) / 2.)
                .collect();
            Sample {
                id: self.id,
                signal: [left, right],
                state: None,
                seed: self.state.map(|(x, _)| x),
                gain: (self.gain + rhs.gain) / 2.,
            }
        } else {
            rhs.add(self)
        }
    }

    fn append(&self, mut other: Sample) -> Self {
        let mut out = self.clone();
        out.signal[0].append(&mut other.signal[0]);
        out.signal[1].append(&mut other.signal[1]);
        out.state = None;
        out
    }

    fn truncate(&self, len: usize) -> Self {
        Sample {
            id: self.id,
            signal: [
                self.signal[0][..len.min(self.len())].to_vec(),
                self.signal[1][..len.min(self.len())].to_vec(),
            ],
            state: None,
            seed: self.state.map(|(x, _)| x),
            gain: self.gain,
        }
    }

    fn skip_silence(&self) -> Self {
        let mut left = vec![];
        let mut right = vec![];

        let mut start = 0;
        let mut end = 0;
        for (i, (l, r)) in self.signal[0].iter().zip(self.signal[1].iter()).enumerate() {
            if l.abs() > f32::EPSILON || r.abs() > f32::EPSILON {
                if i - end > secs_to_frames(SILENCE_TOLERANCE) {
                    left.extend_from_slice(&self.signal[0][start..end]);
                    right.extend_from_slice(&self.signal[1][start..end]);
                    start = i;
                }
                end = i;
            }
        }

        left.extend_from_slice(&self.signal[0][start..end]);
        right.extend_from_slice(&self.signal[1][start..end]);

        Sample {
            id: self.id,
            signal: [left, right],
            state: None,
            seed: self.state.map(|(x, _)| x),
            gain: self.gain,
        }
    }

    fn xcorr(&mut self, planner: &Mutex<RealFftPlanner<f32>>, target: &Sample, ignore_seed: bool) {
        // xcorr = ifft(fft(target) * conj(fft(self)))

        if self.state.is_some() {
            return;
        }

        // select smaller slice of target if seeded
        let seed_region = if ignore_seed {
            0..target.len()
        } else {
            self.seed
                .map(|x| {
                    (x as usize).saturating_sub(self.len())
                        ..(x as usize + 2 * self.len()).min(target.len())
                })
                .unwrap_or(0..target.len())
        };

        // pad both signals
        let padding = vec![0.; (seed_region.len() + self.len() - 1).next_power_of_two()];
        let mut target_left = padding.clone();
        target_left[..seed_region.len()].copy_from_slice(&target.signal[0][seed_region.clone()]);
        let mut target_right = padding.clone();
        target_right[..seed_region.len()].copy_from_slice(&target.signal[1][seed_region.clone()]);
        let mut left = padding.clone();
        left[..self.len()].copy_from_slice(&self.signal[0]);
        let mut right = padding.clone();
        right[..self.len()].copy_from_slice(&self.signal[1]);

        let fft = planner
            .lock()
            .expect("lock fft planner")
            .plan_fft_forward(padding.len());

        // fft(target)
        let mut target_left_fft = fft.make_output_vec();
        let mut target_right_fft = fft.make_output_vec();
        fft.process(&mut target_left, &mut target_left_fft)
            .expect("fft");
        fft.process(&mut target_right, &mut target_right_fft)
            .expect("fft");

        std::mem::drop(target_left);
        std::mem::drop(target_right);

        // fft(self)
        let mut left_fft = fft.make_output_vec();
        let mut right_fft = fft.make_output_vec();
        fft.process(&mut left, &mut left_fft).expect("fft");
        fft.process(&mut right, &mut right_fft).expect("fft");

        std::mem::drop(left);
        std::mem::drop(right);

        // conj([fft(self)]) * [fft(target)]
        left_fft
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = x.conj() * target_left_fft[i]);
        right_fft
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = x.conj() * target_right_fft[i]);

        // ifft([conj(fft(self)) * fft(target)])
        let ifft = planner
            .lock()
            .expect("lock fft planner")
            .plan_fft_inverse(padding.len());
        let mut left_ifft = ifft.make_output_vec();
        let mut right_ifft = ifft.make_output_vec();
        ifft.process(&mut left_fft, &mut left_ifft).expect("ifft");
        ifft.process(&mut right_fft, &mut right_ifft).expect("ifft");

        std::mem::drop(left_fft);
        std::mem::drop(right_fft);

        // find delay and value (signal dot product) of peak
        let (l_delay, l_dot) = left_ifft[..seed_region.len()]
            .iter()
            .enumerate()
            .par_bridge()
            .max_by(|(_, a), (_, b)| a.partial_cmp(&b).unwrap())
            .unwrap();
        let (r_delay, r_dot) = right_ifft[..seed_region.len()]
            .iter()
            .enumerate()
            .par_bridge()
            .max_by(|(_, a), (_, b)| a.partial_cmp(&b).unwrap())
            .unwrap();
        let l_delay = (seed_region.start + l_delay).clamp(0, target.len());
        let r_delay = (seed_region.start + r_delay).clamp(0, target.len());

        self.state = Some((((l_delay + r_delay) / 2) as _, (l_dot + r_dot) / 2.));
    }
}

fn range_intersects(a: &Range<usize>, b: &Range<usize>) -> bool {
    a.contains(&b.start.saturating_sub(1))
        || a.contains(&(b.end + 1))
        || b.contains(&a.start.saturating_sub(1))
        || b.contains(&(a.end + 1))
}

fn range_merge(ranges: &mut Vec<Range<usize>>) {
    ranges.sort_unstable_by_key(|range| range.start);
    let mut out: Vec<Range<usize>> = vec![];
    for range in ranges.iter() {
        if let Some(last) = out.last_mut() {
            if range_intersects(last, range) {
                last.end = last.end.max(range.end);
                continue;
            }
        }
        out.push(range.clone());
    }
    *ranges = out;
}

fn range_overlap_perc(test: &Range<usize>, ranges: &[Range<usize>]) -> f32 {
    let mut acc = 0;
    for range in ranges {
        if range_intersects(test, range) {
            acc += test.end.min(range.end) - test.start.max(range.start);
        }
        if range.start > test.end {
            break;
        }
    }
    acc as f32 / test.len() as f32
}

fn main() {
    assert!(TARGET_CHUNK_SECS > SAMPLE_CHUNK_SECS);

    // setup logging
    fern::Dispatch::new()
        .level(log::LevelFilter::Debug)
        .chain(
            std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open("gen.log")
                .expect("log file"),
        )
        .apply()
        .expect("log");

    let target = Sample::from_wav("target.wav");
    let planner = Mutex::new(RealFftPlanner::new());

    let mut initial = read_dir("samples")
        .expect("read samples folder")
        .filter_map(|file| file.ok())
        .filter(|file| {
            file.file_type().unwrap().is_file() && file.path().extension().unwrap() == "wav"
        })
        .par_bridge()
        .flat_map(|file| {
            Sample::from_wav(file.path())
                .skip_silence()
                .chunked(secs_to_frames(SAMPLE_CHUNK_SECS))
        })
        .collect::<Vec<_>>();
    initial
        .iter_mut()
        .enumerate()
        .for_each(|(i, sample)| sample.id = i as _);

    let out = target
        .chunked(secs_to_frames(TARGET_CHUNK_SECS))
        .into_iter()
        .enumerate()
        .map(|(i, target_chunk)| {
            println!("Section {i}");
            log::debug!("[[Section {}]]", i);

            let mut all = initial.clone();
            let mut out = Sample::new(target_chunk.len());
            let mut out_mask = vec![];

            for j in 0..GENERATIONS {
                log::debug!("[Generation {}]", j);
                log::debug!("{} samples", all.len());

                // calculate and sort by fitness (descending)
                all.par_iter_mut().for_each(|s| {
                    s.xcorr(&planner, &target_chunk, j % SEED_RESET_MOD == 0);

                    // scale by how much it covers untouched regions of output
                    let delay = (s.state.unwrap().0 as usize).min(target_chunk.len() - s.len());
                    s.state.as_mut().unwrap().1 /=
                        range_overlap_perc(&(delay..delay + s.len()), &out_mask).max(0.005) * 2.;
                });
                all.par_sort_unstable_by(|a, b| {
                    b.state.unwrap().1.partial_cmp(&a.state.unwrap().1).unwrap()
                });

                // take the best N samples and add to output
                let mut out_count = 0;
                let mut seen_delays: Vec<usize> = vec![];
                for best in &all {
                    if out_count == OUTPUT_CAPACITY {
                        break;
                    }

                    let best_delay =
                        (best.state.unwrap().0 as usize).min(target_chunk.len() - best.len());

                    // skip samples too close to other samples in this generation
                    let mut skip = false;
                    for seen in &seen_delays {
                        if seen.abs_diff(best_delay) < secs_to_frames(NEARBY_DELAY_TOLERANCE) {
                            skip = true;
                            break;
                        }
                    }
                    if skip {
                        continue;
                    }
                    seen_delays.push(best_delay);
                    out_count += 1;

                    out.signal[0][best_delay..best_delay + best.len()]
                        .par_iter_mut()
                        .zip(best.signal[0].par_iter())
                        .for_each(|(dst, src)| *dst = (*dst + *src) / 2.);
                    out.signal[1][best_delay..best_delay + best.len()]
                        .par_iter_mut()
                        .zip(best.signal[1].par_iter())
                        .for_each(|(dst, src)| *dst = (*dst + *src) / 2.);
                    out_mask.push(best_delay..best_delay + best.len());

                    log::debug!(
                        "Output id {} with delay {}s, {}s",
                        best.id,
                        frames_to_secs(best_delay),
                        frames_to_secs(best.len())
                    );
                }

                range_merge(&mut out_mask);

                // eliminate the weakest
                let survival_count = ((all.len() as f32 * SURVIVAL_RATIO) as usize)
                    .min(all.len())
                    .min(GENERATION_CAPACITY);
                all = all[..survival_count].to_vec();
                log::debug!("{} samples survived", survival_count);

                // transform
                let mut max_len = 0;
                let mut next = vec![];
                all.iter().for_each(|sample| {
                    sample.transforms(
                        OFFSPRINGS_PER_SAMPLE,
                        &target_chunk,
                        &mut next,
                        &mut max_len,
                    );
                });

                let mutation_count = next.len();
                log::debug!("{} mutations generated", next.len());

                // crossovers
                let mut rng = rand::thread_rng();
                let crossover_count = (all.len() as f32 * CROSSOVER_RATIO) as usize;
                let crossover_dist = Uniform::new(0, all.len());
                next.extend(
                    crossover_dist
                        .sample_iter(&mut rng)
                        .take(crossover_count)
                        .map(|k| all[k].add(&all[crossover_dist.sample(&mut rand::thread_rng())])),
                );

                log::debug!("{} crossovers generated", next.len() - mutation_count);

                // elitism
                let elite_count = (all.len() as f32
                    * ELITISM_RATIO
                    * 2f32.powf(ELITISM_COEFF * (j as f32 / GENERATIONS as f32) - ELITISM_COEFF)
                        as f32) as usize;
                let elitism_dist = Triangular::new(0.0f32, all.len() as f32, 0.0f32)
                    .expect("triangular distribution");
                next.extend(
                    elitism_dist
                        .sample_iter(&mut rng)
                        .take(elite_count)
                        .map(|k| all[k as usize].clone()),
                );

                log::debug!("{} samples kept as elites", elite_count);
                log::debug!("");

                all = next;
            }

            out.save_to_wav(format!("out_s{i}.wav"));
            out
        })
        .fold(Sample::new(0), |out, chunk| out.append(chunk));

    out.save_to_wav("out.wav");
}
