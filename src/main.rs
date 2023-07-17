use itertools::Itertools;
use rand_distr::{Distribution, Normal, Triangular, Uniform};
use rayon::{
    prelude::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
        IntoParallelRefMutIterator, ParallelBridge, ParallelIterator,
    },
    slice::ParallelSliceMut,
};
use rubato::{FftFixedInOut, Resampler};
use std::{
    fs::{read_dir, File},
    path::Path,
};

// Sampling rate of the audio while the genetic algorithm runs.
// Lower values will be much faster but will yield lower final audio fidelity.
const HZ: u32 = 20000;
// Minimum duration of a sample.
const SAMPLE_MIN_SECS: f32 = 0.2;
// How many generations will run. More generations means reaching a better fitting solution.
const GENERATIONS: u32 = 500;
// Each N generations the initial population will be mixed in to add variety.
const SOFT_RESET_GENERATION: u32 = 25;
// Each N generations the population is reset to the initial population to escape local optima.
const HARD_RESET_GENERATION: u32 = 100;
// This is how many offspring each sample will create (not taking into account two offspring per one split transformation).
const OFFSPRINGS_PER_SAMPLE: usize = 5;
// The max number of samples that can ever survive from one generation. Take this as the overpopulation threshold.
const GENERATION_CAPACITY: usize = 1000;
// The top N samples are added to the output signal.
const OUTPUT_CAPACITY: usize = 10;
// Ratio of mix between current output and selected samples. 0.5 means each selected sample is average with the current output signal.
const OUTPUT_MIX: f32 = 0.9;
// The top percentile of population that survive to transform/crossover.
const SURVIVAL_RATIO: f32 = 0.5;
// Percentage of surviving population that is chosen (at random) to be crossed over (signals added together).
const CROSSOVER_RATIO: f32 = 0.1;
// What is the ideal duration of a sample?
// Samples deviating from this duration (in seconds) will have their score scaled down.
const TARGET_SAMPLE_DURATION: f32 = 0.7;

// The number of elites selected per generation is calculated as
//      population_count * ELITISM_RATIO * 2^(ELITISM_COEFF * (generation / total_generations) - ELITISM_COEFF)
// This is so that it can "ramp up" as we approach the end of our generation.

// Thew highest percentage of elitism possible.
// Essentially, on the very last generation, this is the percent of the population that will be taken as elites.
// It's recommended that if you're running a low number of generations, that this be equal or very close to 0.
const ELITISM_RATIO: f32 = 0.1;
// How much the elitism is scaled through the generations.
// The higher this number is, the more delayed elitism will be.
// Must be >= 0. To disable "dynamic elitism", set this value to 0 and the elitism ratio will remain constant through every generation.
const ELITISM_COEFF: f32 = 10.0;

// Number of bins of the spectrogram on the x-axis.
// Longer target audios may need more bins for better time resolution.
const SPECT_XBINS: usize = 2048;
// Number of bins of the spectrogram on the y-axis
const SPECT_YBINS: usize = 256;

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
    pub state: Option<(u32, f64)>,
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

        let left = self.signal[0].par_iter().map(|x| x * ratio).collect();
        let right = self.signal[1].par_iter().map(|x| x * ratio).collect();
        Sample {
            id: self.id,
            signal: [left, right],
            state: None,
            gain: self.gain * ratio,
            spect: None,
        }
    }

    fn transforms(&self, n: usize, target: &Sample, out: &mut Vec<Sample>, max_len: &mut usize) {
        let type_dist = Uniform::new_inclusive(0, 4);
        let split_dist = Uniform::new(self.len() / 4, 3 * self.len() / 4);
        let pitch_dist = Uniform::new_inclusive(-200, 200);
        let amplitude_dist = Normal::new(1.0f32, 0.2).expect("normal distribution");
        let duplicate_dist = Uniform::new_inclusive(1, 5);
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
            _ => unreachable!(),
        });
    }

    fn add(&self, rhs: &Sample, delay: usize, mix: f32) -> Self {
        if self.len() >= rhs.len() {
            let left = self.signal[0]
                .par_iter()
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
                .par_iter()
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
            rhs.add(self, 0, 1. - mix)
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

    fn compute_spect(&mut self, max: usize) {
        if self.spect.is_some() {
            return;
        }

        let xbins = ((self.len() as f32 / max as f32) * SPECT_XBINS as f32) as usize;

        self.spect = Some(
            sonogram::SpecOptionsBuilder::new(xbins)
                .load_data_from_memory_f32(self.signal[0].clone(), HZ)
                .downsample(4)
                .set_window_fn(sonogram::hann_function)
                .build()
                .expect("spectrogram")
                .compute()
                .to_buffer(sonogram::FrequencyScale::Log, xbins, SPECT_YBINS),
        );
    }

    fn mse(&mut self, curr: &Sample, target: &Sample) {
        // mse = 1/n * (target - self)^2

        let delay =
            Uniform::new(0, (target.len() - self.len()).max(1)).sample(&mut rand::thread_rng());

        let mut next = curr.add(self, delay, OUTPUT_MIX);
        next.compute_spect(target.len());
        let mut sum = target
            .spect
            .as_ref()
            .unwrap()
            .iter()
            .zip(next.spect.as_ref().unwrap())
            .map(|(a, b)| ((a - b) * (a - b)) as f64)
            .sum::<f64>();

        sum *= (1.
            + (frames_to_secs(self.len()) - TARGET_SAMPLE_DURATION).abs() / TARGET_SAMPLE_DURATION)
            as f64;

        self.state = Some((delay as _, sum));
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
    target.compute_spect(target.len());

    let mut initial = read_dir("samples")
        .expect("read samples folder")
        .filter_map(|file| file.ok())
        .filter(|file| {
            file.file_type().unwrap().is_file() && file.path().extension().unwrap() == "wav"
        })
        .par_bridge()
        .flat_map(|file| Sample::from_wav(file.path()).chunked(target.len()))
        .collect::<Vec<_>>();
    initial
        .iter_mut()
        .enumerate()
        .for_each(|(i, sample)| sample.id = i as _);

    let mut all = initial.clone();
    let mut out = Sample::new(target.len());

    let _ = std::fs::remove_dir_all("checkpoints");
    std::fs::create_dir("checkpoints").expect("mkdir checkpoints");

    for j in 0..GENERATIONS {
        println!("Generation {}", j + 1);
        log::debug!("[Generation {}]", j + 1);

        if (j + 1) % HARD_RESET_GENERATION == 0 {
            all = initial.clone();
        } else if (j + 1) % SOFT_RESET_GENERATION == 0 {
            all.append(&mut initial.clone());
        }

        log::debug!("{} samples", all.len());

        // calculate and sort by fitness
        all = all
            .into_par_iter()
            .filter(|s| s.len() >= secs_to_frames(SAMPLE_MIN_SECS))
            .collect();
        all.par_iter_mut().for_each(|s| {
            s.mse(&out, &target);
        });
        all.par_sort_unstable_by(|a, b| {
            a.state.unwrap().1.partial_cmp(&b.state.unwrap().1).unwrap()
        });

        // keep the best
        all.iter()
            .take(OUTPUT_CAPACITY.min(all.len()))
            .rev()
            .for_each(|best| {
                let best_delay = (best.state.unwrap().0 as usize).min(target.len() - best.len());
                out = out.add(best, best_delay, OUTPUT_MIX);

                log::debug!(
                    "Output id {} with delay {}s, {}s",
                    best.id,
                    frames_to_secs(best_delay),
                    frames_to_secs(best.len())
                );
            });

        if (j + 1) % 10 == 0 {
            out.save_to_wav(format!("checkpoints/G{}.wav", j + 1));
        }

        log::debug!(
            "Best score: {}, worst score: {}",
            all[0].state.unwrap().1,
            all.last().unwrap().state.unwrap().1
        );

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
                        0 => all[k].add(&all[crossover_dist.sample(&mut rng)], 0, 0.5),
                        1 => all[k].append(&all[crossover_dist.sample(&mut rng)]),
                        _ => unreachable!(),
                    }
                }),
        );

        log::debug!("{} crossovers generated", next.len() - mutation_count);

        // elitism
        let elite_count = (all.len() as f32
            * ELITISM_RATIO
            * 2f32.powf(ELITISM_COEFF * (j as f32 / GENERATIONS as f32) - ELITISM_COEFF) as f32)
            as usize;
        let elitism_dist =
            Triangular::new(0.0f32, all.len() as f32, 0.0f32).expect("triangular distribution");
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

    out.save_to_wav("out.wav");
}
