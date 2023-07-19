use itertools::Itertools;
use rand_distr::{Distribution, Normal, Uniform};
use rubato::{FftFixedInOut, Resampler};
use std::{
    fs::{read_dir, File},
    io::Write,
    path::Path,
};

// Sampling rate of the audio while the genetic algorithm runs.
// Lower values will be much faster but will yield lower final audio fidelity.
const HZ: u32 = 48000;
// How many cycles will run.
const CYCLES: u32 = 500;
// The min number of generations that will run per cycle.
const GENERATION_MIN: u32 = 1;
// The max number of generations that will run per cycle. This prevents run-off cycles that dig themselves into a hole of worse-than-before solutions.
const GENERATION_MAX: u32 = 7;
// This is how many offspring each sample will create (not taking into account two offspring per one split transformation).
const OFFSPRINGS_PER_SAMPLE: usize = 5;
// The max number of samples that can ever survive from one generation. Take this as the overpopulation threshold.
const GENERATION_CAPACITY: usize = 500;
// The top N samples are added to the output signal.
const OUTPUT_CAPACITY: usize = 10;
// The top percentile of population that survive to transform/crossover.
const SURVIVAL_RATIO: f32 = 0.3;
// Percentage of surviving population that is chosen (at random) to be crossed over (signals added together).
const CROSSOVER_RATIO: f32 = 0.1;
// What is the BPM of the target?
const TARGET_BPM: f32 = 115.0;
// 0-1, amount of rhythm
const I_GOT_RHYTHM: f32 = 1.0;

// Number of bins of the spectrogram on the x-axis.
// Longer target audios may need more bins for better time resolution.
const SPECT_XBINS: usize = 256;
// Number of bins of the spectrogram on the y-axis
const SPECT_YBINS: usize = 64;

fn freq_pitch(cents: i32) -> u32 {
    (2.0f32.powf(cents as f32 / 1200.0) * HZ as f32) as _
}

fn secs_to_frames(secs: f32) -> usize {
    (HZ as f32 * secs) as _
}

fn frames_to_secs(frames: usize) -> f32 {
    frames as f32 / HZ as f32
}

fn bpm_to_frames(bpm: f32) -> usize {
    secs_to_frames(60. / bpm)
}

#[derive(Clone)]
struct Sample {
    pub id: u64,
    pub signal: [Vec<f32>; 2], // stereo
    pub state: Option<(u32, f64, f32)>,
    pub gain: f32,
    pub spect: Option<Vec<f32>>,
}

impl Sample {
    fn new(len: usize) -> Self {
        Sample {
            id: 0,
            signal: [vec![0.; len], vec![0.; len]],
            state: None,
            gain: 1.,
            spect: None,
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
            wav::BitDepth::Eight(samples) => samples
                .into_iter()
                .map(|sample| sample as f32 / 0xFF as f32)
                .collect::<Vec<_>>(),
            wav::BitDepth::ThirtyTwoFloat(samples) => samples,
            _ => unimplemented!(),
        };
        let (l, r) = interleaved.chunks(2).map(|x| (x[0], x[1])).unzip();
        Sample {
            id: 0,
            signal: [l, r],
            state: None,
            gain: 1.,
            spect: None,
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
                gain: self.gain,
                spect: None,
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
        let mut left_out = vec![];
        left_out.reserve(resampler.output_frames_max());
        let mut right_out = vec![];
        right_out.reserve(resampler.output_frames_max());

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
            gain: self.gain,
            spect: None,
        }
    }

    fn reverse(&self) -> Self {
        let mut out = self.signal.clone();
        out.reverse();
        Sample {
            id: self.id,
            signal: out,
            state: None,
            gain: self.gain,
            spect: None,
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
                gain: self.gain,
                spect: None,
            },
            Sample {
                id: self.id,
                signal: [b_left.to_vec(), b_right.to_vec()],
                state: None,
                gain: self.gain,
                spect: None,
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
                gain: self.gain,
                spect: None,
            };
        }

        let left = self.signal[0].iter().map(|x| x * ratio).collect();
        let right = self.signal[1].iter().map(|x| x * ratio).collect();
        Sample {
            id: self.id,
            signal: [left, right],
            state: None,
            gain: self.gain * ratio,
            spect: None,
        }
    }

    fn transforms(&self, n: usize, target: &Sample, out: &mut Vec<Sample>, max_len: &mut usize) {
        let type_dist = Uniform::new_inclusive(0, 5);
        let split_dist = Uniform::new(self.len() / 4, 3 * self.len() / 4);
        let pitch_dist = Uniform::new_inclusive(-200, 200);
        let amplitude_dist = Normal::new(1.0f32, 0.2).expect("normal distribution");
        let duplicate_dist = Uniform::new_inclusive(1, 3);
        let chunk_dist = Uniform::new_inclusive(self.len() / 8, self.len() / 3);
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
                out.push(self.amplitude(amplitude_dist.sample(&mut rng).clamp(0.6, 1.5)));
                *max_len = (*max_len).max(self.len());
            }
            3 => {
                out.push(
                    self.pitch(pitch_dist.sample(&mut rng))
                        .truncate(target.len()),
                );
                *max_len = (*max_len).max(self.len());
            }
            4 => out.append(&mut vec![self.duplicate(); duplicate_dist.sample(&mut rng)]),
            5 => out.append(
                &mut self.chunked(
                    chunk_dist
                        .sample(&mut rng)
                        .max(secs_to_frames(0.2).min(self.len())),
                ),
            ),
            _ => unreachable!(),
        });
    }

    fn add(&self, rhs: &Sample, delay: usize, mix: f32, trunc: bool) -> Self {
        if self.len() >= rhs.len() || trunc {
            let left = self.signal[0]
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    if i >= delay && i < delay + rhs.len() {
                        (1. - mix) * x + mix * rhs.signal[0][i - delay]
                    } else {
                        *x
                    }
                })
                .collect();
            let right = self.signal[1]
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    if i >= delay && i < delay + rhs.len() {
                        (1. - mix) * x + mix * rhs.signal[1][i - delay]
                    } else {
                        *x
                    }
                })
                .collect();
            Sample {
                id: self.id,
                signal: [left, right],
                state: None,
                gain: (1. - mix) * self.gain + mix * rhs.gain,
                spect: None,
            }
        } else {
            rhs.add(self, 0, 1. - mix, true)
        }
    }

    fn append(&self, other: &Sample) -> Self {
        let mut l = self.signal[0].clone();
        l.append(&mut other.signal[0].clone());
        let mut r = self.signal[1].clone();
        r.append(&mut other.signal[1].clone());
        Sample {
            id: self.id,
            signal: [l, r],
            state: None,
            gain: (self.gain + other.gain) / 2.,
            spect: None,
        }
    }

    fn truncate(&self, len: usize) -> Self {
        Sample {
            id: self.id,
            signal: [
                self.signal[0][..len.min(self.len())].to_vec(),
                self.signal[1][..len.min(self.len())].to_vec(),
            ],
            state: None,
            gain: self.gain,
            spect: None,
        }
    }

    fn duplicate(&self) -> Self {
        Sample {
            id: self.id,
            signal: self.signal.clone(),
            state: None,
            gain: self.gain,
            spect: None,
        }
    }

    fn compute_spect(&mut self) {
        if self.spect.is_some() {
            return;
        }

        self.spect = Some(
            sonogram::SpecOptionsBuilder::new(SPECT_XBINS)
                .load_data_from_memory_f32(self.signal[0].clone(), HZ)
                .downsample(4)
                .set_window_fn(sonogram::hann_function)
                .build()
                .expect("spectrogram")
                .compute()
                .to_buffer(sonogram::FrequencyScale::Log, SPECT_XBINS, SPECT_YBINS),
        );
    }

    fn mse(a: &Sample, b: &Sample) -> f64 {
        a.spect
            .as_ref()
            .unwrap()
            .iter()
            .zip(b.spect.as_ref().unwrap())
            .map(|(a, b)| ((a - b) * (a - b)) as f64)
            .sum::<f64>()
    }

    fn compute_state(&mut self, curr: &Sample, target: &Sample, delay: Option<usize>) {
        // mse = 1/n * (target - self)^2

        let delay = delay.unwrap_or_else(|| {
            let pick = Uniform::new(0, (target.len() - self.len()).max(1))
                .sample(&mut rand::thread_rng()) as f64;
            let on_time = (pick as f64 / bpm_to_frames(TARGET_BPM) as f64).round()
                * bpm_to_frames(TARGET_BPM) as f64;
            (pick * (1. - I_GOT_RHYTHM as f64) + on_time * I_GOT_RHYTHM as f64) as _
        });

        let mix = Uniform::new(0.5, 1.0).sample(&mut rand::thread_rng());

        let mut next = curr.add(self, delay, mix, true);
        next.compute_spect();
        let sum = Sample::mse(&next, &target);

        self.state = Some((delay as _, sum, mix));
    }
}

fn main() {
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

    let mut target = Sample::from_wav("target.wav");
    target.compute_spect();

    let mut initial = read_dir("samples")
        .expect("read samples folder")
        .filter_map(|file| file.ok())
        .filter(|file| {
            file.file_type().unwrap().is_file() && file.path().extension().unwrap() == "wav"
        })
        .flat_map(|file| Sample::from_wav(file.path()).chunked(target.len()))
        .collect::<Vec<_>>();
    initial
        .iter_mut()
        .enumerate()
        .for_each(|(i, sample)| sample.id = i as _);

    let mut out = Sample::new(target.len());

    let _ = std::fs::remove_dir_all("checkpoints");
    std::fs::create_dir("checkpoints").expect("mkdir checkpoints");

    out.compute_spect();
    let mut last_score = Sample::mse(&out, &target);
    let first_score = last_score;

    let mut max_cycles = CYCLES;
    for i in 0..max_cycles {
        println!("Cycle {}", i + 1);
        log::debug!("[Cycle {}]", i + 1);

        let sort = |mut all: Vec<Sample>| {
            // calculate and sort by fitness
            all = all
                .into_iter()
                .filter(|s| s.len() >= secs_to_frames(0.2))
                .collect();
            all.iter_mut().for_each(|s| {
                s.compute_state(&out, &target, None);
            });
            all.sort_unstable_by(|a, b| {
                a.state.unwrap().1.partial_cmp(&b.state.unwrap().1).unwrap()
            });
            all
        };

        let eliminate = |mut all: Vec<Sample>| {
            // eliminate the weakest
            let survival_count = ((all.len() as f32 * SURVIVAL_RATIO) as usize)
                .min(all.len())
                .min(GENERATION_CAPACITY);
            all = all[..survival_count].to_vec();
            log::debug!("{} samples survived", survival_count);
            all
        };

        let mutate = |mut all: Vec<Sample>| {
            // transform
            let mut max_len = 0;
            let mut next = vec![];
            all.iter().for_each(|sample| {
                sample.transforms(OFFSPRINGS_PER_SAMPLE, &target, &mut next, &mut max_len);
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
                    .map(|k| {
                        let mut rng = rand::thread_rng();
                        match Uniform::new(0, 2).sample(&mut rng) {
                            0 => all[k].add(&all[crossover_dist.sample(&mut rng)], 0, 0.5, false),
                            1 => all[k].append(&all[crossover_dist.sample(&mut rng)]),
                            _ => unreachable!(),
                        }
                    }),
            );

            log::debug!("{} crossovers generated", next.len() - mutation_count);
            log::debug!("");

            all.append(&mut next);
            all
        };

        let write = |all: &[Sample], out: &mut Sample, write_log: bool| {
            all.iter()
                .take(OUTPUT_CAPACITY.min(all.len()))
                .rev()
                .for_each(|best| {
                    let (delay, mix) = best
                        .state
                        .map(|(delay, _, mix)| {
                            ((delay as usize).min(target.len() - best.len()), mix)
                        })
                        .unwrap();
                    *out = out.add(best, delay, mix, true);

                    if write_log {
                        log::debug!(
                            "Output id {} with delay {}s, {}s, mix {}",
                            best.id,
                            frames_to_secs(delay),
                            frames_to_secs(best.len()),
                            mix,
                        );
                    }
                });
        };

        let mut all = initial.clone();

        all = mutate(all);
        let mut best_score;
        let mut j = 0;
        while {
            print!("Gen {}, ", j + 1);
            std::io::stdout().flush().unwrap();
            log::debug!("[Generation {}]", j + 1);
            log::debug!("{} samples", all.len());

            all = eliminate(sort(all));

            let mut scored = out.clone();
            write(&all, &mut scored, false);
            scored.compute_spect();
            best_score = Sample::mse(&scored, &target);

            all = mutate(all);

            j += 1;
            j < GENERATION_MIN || (j < GENERATION_MAX && best_score > last_score)
        } {}

        if best_score > last_score {
            println!("... Regressed\n");
            max_cycles += 1;
            continue;
        }

        all = sort(all);
        println!();

        // keep the best
        write(&all, &mut out, true);

        out.spect = None;
        out.compute_spect();
        last_score = Sample::mse(&out, &target);

        println!("{:.2}%", 100. * (1. - last_score / first_score));

        if (i + 1) % 10 == 0 {
            out.save_to_wav(format!("checkpoints/C{}.wav", i + 1));
        }

        log::debug!(
            "Best score: {}, worst score: {}",
            all[0].state.unwrap().1,
            all.last().unwrap().state.unwrap().1
        );
    }

    out.save_to_wav("out.wav");
}
