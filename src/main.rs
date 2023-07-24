mod stereo;

use rand_distr::{Distribution, Normal, Uniform};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};
use regex::Regex;
use std::{fs::read_dir, io::Write, path::Path, rc::Rc};
use stereo::*;

// Sampling rate of the audio while the genetic algorithm runs.
// Lower values will be much faster but will yield lower final audio fidelity.
const HZ: u32 = 44100;
// How many cycles will run.
const CYCLES: u32 = 500;
// The min number of generations that will run per cycle.
const GENERATION_MIN: u32 = 1;
// The max number of generations that will run per cycle. This prevents run-off cycles that dig themselves into a hole of worse-than-before solutions.
const GENERATION_MAX: u32 = 6;
// Initial population size per cycle, chosen at random from the loaded samples.
const INITIAL_POPULATION: usize = 100;
// This is how many offspring each sample will create (not taking into account two offspring per one split transformation).
const OFFSPRINGS_PER_SAMPLE: usize = 5;
// The max number of samples that can ever survive from one generation. Take this as the overpopulation threshold.
const GENERATION_CAPACITY: usize = 500;
// The top N samples are added to the output signal.
const OUTPUT_CAPACITY: usize = 10;
// Percentage of surviving population that is chosen (at random) to be crossed over (signals added together).
const CROSSOVER_RATIO: f32 = 0.2;
// The top percentile of population that survive to transform/crossover.
const SURVIVAL_RATIO: f32 = 0.4;
// 0-1, amount of rhythm.
const I_GOT_RHYTHM: f32 = 0.8;
// What is the BPM of the target?
const TARGET_BPM: f32 = 160.;
// Should very short samples be filtered from the population? Disable if doing a percussion pass.
const FILTER_SHORT: bool = false;
// Should regression on subsequent cycles be permitted (once the generation max is hit)?
const ALLOW_REGRESSION: bool = false;

// Number of bins of the spectrogram on the x-axis.
// Longer target audios may need more bins for better time resolution.
const SPECT_XBINS: usize = 1600;
// Number of bins of the spectrogram on the y-axis
const SPECT_YBINS: usize = 256;

fn secs_to_frames(secs: f32) -> usize {
    (HZ as f32 * secs) as _
}

fn bpm_to_frames(bpm: f32) -> usize {
    secs_to_frames(60. / bpm)
}

#[derive(Clone)]
enum Fx<'a> {
    Reverse,
    Gain(f32),
    Pitch(i32),
    Add(Sample<'a>, usize, f32),
    Append(Sample<'a>),
    Repeat(usize),
    Shelf { low: bool, gain: f32, cutoff: u32 },
}

#[derive(Clone)]
struct Sample<'a> {
    pub id: u64,
    pub signal: StereoRef<'a>,
    pub fx: Vec<Fx<'a>>,
    pub state: Option<(u32, f32, f32)>,
}

impl<'a> Sample<'a> {
    fn new(signal: StereoRef<'a>) -> Self {
        Sample {
            id: 0,
            signal,
            fx: vec![],
            state: None,
        }
    }

    fn len(&self) -> usize {
        self.signal[0].len()
    }

    fn with_fx(&self, new_fx: Fx<'a>) -> Self {
        let mut fx = self.fx.clone();
        fx.push(new_fx);
        Sample {
            id: self.id,
            signal: self.signal.clone(),
            fx,
            state: None,
        }
    }

    fn split(&self, at: usize) -> [Self; 2] {
        let [a, b] = split(self.signal, at);
        [
            Sample {
                id: self.id,
                signal: a,
                fx: self.fx.clone(),
                state: None,
            },
            Sample {
                id: self.id,
                signal: b,
                fx: self.fx.clone(),
                state: None,
            },
        ]
    }

    fn chunked(&self, frames: usize) -> Vec<Self> {
        chunked(self.signal, frames)
            .into_iter()
            .map(|signal| Sample {
                id: self.id,
                signal,
                fx: self.fx.clone(),
                state: None,
            })
            .collect()
    }

    fn duplicate(&self) -> Self {
        Sample {
            id: self.id,
            signal: self.signal.clone(),
            fx: self.fx.clone(),
            state: None,
        }
    }

    fn transforms(&self, n: usize, out: &mut Vec<Sample<'a>>) {
        let mut rng = rand::thread_rng();

        let types = [0, 1, 2, 3, 4, 5, 6, 7];
        let type_dist = Uniform::new(0, types.len());

        let split_dist = Uniform::new(self.len() / 4, 3 * self.len() / 4);
        let pitch_dist = Uniform::new_inclusive(-600, 600);
        let amplitude_dist = Normal::new(1.0f32, 0.2).expect("amplitude dist");
        let duplicate_dist = Uniform::new_inclusive(1, 3);
        let chunk_dist = Uniform::new_inclusive(self.len() / 8, self.len() / 3);
        let repeat_dist = Uniform::new(1, 4);
        let shelf_gain = Uniform::new(-30., 30.);
        let shelf_cutoff = Uniform::new_inclusive(800., 1200.);

        (0..n).for_each(|_| match types[type_dist.sample(&mut rng)] {
            0 => {
                out.push(self.with_fx(Fx::Reverse));
            }
            1 => {
                out.extend_from_slice(&self.split(split_dist.sample(&mut rng)));
            }
            2 => {
                out.push(self.with_fx(Fx::Gain(amplitude_dist.sample(&mut rng).clamp(0.6, 1.5))));
            }
            3 => {
                out.push(self.with_fx(Fx::Pitch(pitch_dist.sample(&mut rng))));
            }
            4 => out.append(&mut vec![self.duplicate(); duplicate_dist.sample(&mut rng)]),
            5 => out.append(&mut self.chunked(chunk_dist.sample(&mut rng))),
            6 => out.push(self.with_fx(Fx::Repeat(repeat_dist.sample(&mut rng)))),
            7 => out.push(self.with_fx(Fx::Shelf {
                low: rand::random(),
                gain: shelf_gain.sample(&mut rng),
                cutoff: shelf_cutoff.sample(&mut rng) as _,
            })),
            _ => unreachable!(),
        });
    }

    fn waveform(&self) -> Stereo {
        self.fx
            .iter()
            .fold(stereo_to_owned(self.signal), |signal, fx| match fx {
                Fx::Reverse => reverse(stereo_borrow(&signal)),
                Fx::Gain(ratio) => gain(stereo_borrow(&signal), *ratio),
                Fx::Pitch(cents) => pitch(stereo_borrow(&signal), HZ, *cents),
                Fx::Add(sample, delay, mix) => add(
                    stereo_borrow(&signal),
                    stereo_borrow(&sample.waveform()),
                    *delay,
                    Some(*mix),
                    false,
                ),
                Fx::Append(sample) => {
                    append(stereo_borrow(&signal), stereo_borrow(&sample.waveform()))
                }
                Fx::Repeat(n) => {
                    let mut out = signal.clone();
                    for _ in 0..*n {
                        out = append(stereo_borrow(&out), stereo_borrow(&signal));
                    }
                    out
                }
                Fx::Shelf { low, gain, cutoff } => {
                    shelf(stereo_borrow(&signal), *low, HZ, *cutoff, *gain)
                }
            })
    }

    fn compute_state(
        &mut self,
        waveform: StereoRef,
        curr: StereoRef,
        target_spect: &[f32],
        target_len: usize,
        delay: Option<usize>,
    ) {
        if rms(waveform) < 0.1 {
            self.state = Some((0, f32::INFINITY, 0.));
            return;
        }

        let delay = delay.unwrap_or_else(|| {
            let pick = Uniform::new(0, (target_len - waveform[0].len()).max(1))
                .sample(&mut rand::thread_rng()) as f32;
            let on_time = (pick / bpm_to_frames(TARGET_BPM) as f32).round()
                * bpm_to_frames(TARGET_BPM) as f32;
            (pick * (1. - I_GOT_RHYTHM) + on_time * I_GOT_RHYTHM) as _
        });

        let score = mse(
            &spectro(
                stereo_borrow(&normalize(stereo_borrow(&add(
                    curr, waveform, delay, None, true,
                )))),
                HZ,
                SPECT_XBINS,
                SPECT_YBINS,
            ),
            target_spect,
        );

        self.state = Some((delay as _, score, 1.0));
    }
}

fn mse(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / a.len() as f32
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

    let target_signal = load_wav("target.wav", HZ);
    let target_spect = spectro(stereo_borrow(&target_signal), HZ, SPECT_XBINS, SPECT_YBINS);

    let mut cache = vec![];

    let mut initial = read_dir("samples")
        .expect("read samples folder")
        .filter_map(|file| file.ok())
        .filter(|file| {
            file.file_type().unwrap().is_file() && file.path().extension().unwrap() == "wav"
        })
        .flat_map(|file| {
            let signal = Rc::new(load_wav(file.path(), HZ));
            cache.push(signal.clone());
            Sample::new(unsafe {
                [
                    std::mem::transmute(&signal[0][..]),
                    std::mem::transmute(&signal[0][..]),
                ]
            })
            .chunked(8 * bpm_to_frames(TARGET_BPM))
        })
        //.filter(|chunk| rms(chunk.signal) > 0.1)
        .collect::<Vec<_>>();
    initial
        .iter_mut()
        .enumerate()
        .for_each(|(i, sample)| sample.id = i as _);

    let mut out = [
        vec![0.; target_signal[0].len()],
        vec![0.; target_signal[0].len()],
    ];

    let base_score = mse(
        &spectro(stereo_borrow(&out), HZ, SPECT_XBINS, SPECT_YBINS),
        &target_spect,
    );

    let mut first_cycle = 0;
    if Path::new("checkpoints").is_dir() {
        let re = Regex::new(r"^C([0-9]+)\.wav$").unwrap();

        if let Some((file, cycle)) = read_dir("checkpoints")
            .expect("read checkpoints folder")
            .filter_map(|file| file.ok())
            .filter_map(|file| {
                file.file_type()
                    .unwrap()
                    .is_file()
                    .then(|| {
                        re.captures(file.file_name().to_str()?)
                            .and_then(|caps| caps.get(1))
                            .and_then(|cap| cap.as_str().parse::<u32>().ok())
                    })
                    .flatten()
                    .map(|cycle| (file, cycle))
            })
            .max_by_key(|(_, cycle)| *cycle)
        {
            first_cycle = cycle;
            out = load_wav(file.path(), HZ);
            println!(
                "Loaded checkpoint {} at cycle {}",
                file.path().file_name().unwrap().to_str().unwrap(),
                cycle,
            );
        }
    } else {
        std::fs::create_dir("checkpoints").expect("mkdir checkpoints");
    }

    let mut last_score = mse(
        &spectro(stereo_borrow(&out), HZ, SPECT_XBINS, SPECT_YBINS),
        &target_spect,
    );

    if first_cycle > 0 {
        println!("{:.2}%", 100. * (1. - last_score / base_score));
    }

    let mut max_cycles = CYCLES;
    let mut i = first_cycle;
    while i < max_cycles {
        println!("Cycle {}", i + 1);
        log::debug!("[Cycle {}]", i + 1);

        fn sort<'a>(
            mut all: Vec<Sample<'a>>,
            out: StereoRef,
            target_spect: &[f32],
            target_len: usize,
        ) -> Vec<Sample<'a>> {
            // calculate and sort by fitness
            all.retain(|s| !FILTER_SHORT || s.len() >= bpm_to_frames(TARGET_BPM));
            all.par_iter_mut().for_each(|s| {
                let waveform = s.waveform();
                s.compute_state(
                    stereo_borrow(&waveform),
                    out,
                    target_spect,
                    target_len,
                    None,
                );
            });
            all.sort_unstable_by(|a, b| {
                a.state.unwrap().1.partial_cmp(&b.state.unwrap().1).unwrap()
            });
            all
        }

        fn eliminate(mut all: Vec<Sample>) -> Vec<Sample> {
            // eliminate the weakest
            let survival_count = ((all.len() as f32 * SURVIVAL_RATIO) as usize)
                .min(all.len())
                .min(GENERATION_CAPACITY);
            all = all[..survival_count].to_vec();
            log::debug!("{} samples survived", survival_count);
            all
        }

        fn mutate(mut all: Vec<Sample>) -> Vec<Sample> {
            // transform
            let mut next = vec![];
            all.iter().for_each(|sample| {
                sample.transforms(OFFSPRINGS_PER_SAMPLE, &mut next);
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
                            0 => all[k].with_fx(Fx::Add(
                                all[crossover_dist.sample(&mut rng)].clone(),
                                0,
                                0.5,
                            )),
                            1 => all[k]
                                .with_fx(Fx::Append(all[crossover_dist.sample(&mut rng)].clone())),
                            _ => unreachable!(),
                        }
                    }),
            );

            log::debug!("{} crossovers generated", next.len() - mutation_count);
            log::debug!("");

            all.append(&mut next);
            all
        }

        fn write(all: &[Sample], out: &mut Stereo, target_len: usize) {
            all.iter()
                .take(OUTPUT_CAPACITY.min(all.len()))
                .rev()
                .for_each(|best| {
                    let (delay, _mix) = best
                        .state
                        .map(|(delay, _, mix)| ((delay as usize).min(target_len - best.len()), mix))
                        .unwrap();

                    *out = add(
                        stereo_borrow(out),
                        stereo_borrow(&best.waveform()),
                        delay,
                        None,
                        true,
                    );
                });

            *out = normalize(stereo_borrow(out));
        }

        let mut all: Vec<_> = Uniform::new(0, initial.len())
            .sample_iter(&mut rand::thread_rng())
            .take(INITIAL_POPULATION.min(initial.len()))
            .map(|i| initial[i].clone())
            .collect();

        all = mutate(all);
        let mut best_score;
        let mut j = 0;
        let mut best_out;
        while {
            print!("Gen {}, ", j + 1);
            std::io::stdout().flush().unwrap();
            log::debug!("[Generation {}]", j + 1);
            log::debug!("{} samples", all.len());

            all = eliminate(sort(
                all,
                stereo_borrow(&out),
                &target_spect,
                target_signal[0].len(),
            ));

            best_out = out.clone();
            write(&all, &mut best_out, target_signal[0].len());
            best_score = mse(
                &spectro(stereo_borrow(&best_out), HZ, SPECT_XBINS, SPECT_YBINS),
                &target_spect,
            );

            all = mutate(all);

            j += 1;
            j < GENERATION_MIN || (j < GENERATION_MAX && best_score > last_score)
        } {}

        if !ALLOW_REGRESSION && best_score > last_score {
            println!("... Regressed\n");
            max_cycles += 1;
            continue;
        }

        all = sort(
            all,
            stereo_borrow(&out),
            &target_spect,
            target_signal[0].len(),
        );
        println!();

        // keep the best
        out = best_out;

        last_score = best_score;

        println!("{:.2}%", 100. * (1. - last_score / base_score));

        if (i + 1) % 10 == 0 {
            save_to_wav(
                stereo_borrow(&out),
                format!("checkpoints/C{}.wav", i + 1),
                HZ,
            );
        }

        log::debug!(
            "Best score: {}, worst score: {}",
            all[0].state.unwrap().1,
            all.last().unwrap().state.unwrap().1
        );

        i += 1;
    }

    save_to_wav(stereo_borrow(&out), "out.wav", HZ);
}
