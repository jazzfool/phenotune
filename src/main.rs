use rayon::{
    prelude::{
        IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelBridge, ParallelIterator,
    },
    slice::ParallelSliceMut,
};
use rubato::{FastFixedIn, Resampler};
use rustfft::{num_complex::Complex32, FftPlanner};
use std::{
    fs::{read_dir, File},
    path::Path,
    sync::Mutex,
};

const HZ: u32 = 44100;
const SAMPLE_CHUNK_SECS: f32 = 5.0;
const CYCLES: u32 = 3;
const GENERATIONS_PER_CYCLE: u32 = 1;
const OFFSPRINGS_PER_SAMPLE: usize = 1;
const SURVIVAL_CAPACITY: usize = 200;

fn freq_pitch(cents: i32) -> u32 {
    (2.0f32.powf(cents as f32 / 1200.0) * HZ as f32) as _
}

fn secs_to_frames(secs: f32) -> usize {
    (HZ as f32 * secs) as _
}

#[derive(Clone)]
struct Sample {
    pub signal: Vec<[f32; 2]>, // stereo
    pub state: Option<(u32, f32)>,
}

impl Sample {
    fn new(len: usize) -> Self {
        Sample {
            signal: vec![[0., 0.]; len],
            state: None,
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
        let signal = interleaved.chunks(2).map(|x| [x[0], x[1]]).collect();
        Sample {
            signal,
            state: None,
        }
        .resample(header.sampling_rate, HZ)
    }

    fn save_to_wav(&self, file: impl AsRef<Path>) {
        let mut file = File::create(file).expect("write file");
        wav::write(
            wav::Header::new(wav::WAV_FORMAT_PCM, 2, HZ, 16),
            &wav::BitDepth::Sixteen(
                self.signal
                    .iter()
                    .flat_map(|&[l, r]| {
                        [
                            (l.clamp(-1., 1.) * 0x8000 as f32) as i16,
                            (r.clamp(-1., 1.) * 0x8000 as f32) as i16,
                        ]
                    })
                    .collect(),
            ),
            &mut file,
        )
        .expect("encode wav");
    }

    fn chunked(&self, frames: usize) -> Vec<Self> {
        self.signal
            .chunks(frames)
            .map(|signal| Sample {
                signal: signal.to_vec(),
                state: None,
            })
            .collect()
    }

    fn resample(&self, f1: u32, f2: u32) -> Self {
        const CHUNK_SIZE: usize = 2048;

        if f1 == f2 {
            return self.clone();
        }

        let (left, right): (Vec<_>, Vec<_>) = self.signal.iter().map(|&[l, r]| (l, r)).unzip();

        let mut resampler = FastFixedIn::new(
            f2 as f64 / f1 as f64,
            1.,
            rubato::PolynomialDegree::Linear,
            CHUNK_SIZE,
            2,
        )
        .expect("init resampler");
        let mut left_out = vec![0.; resampler.output_frames_max()];
        let mut right_out = vec![0.; resampler.output_frames_max()];

        let mut remaining = left.len();
        let mut next_frames = resampler.input_frames_next();
        while remaining > next_frames {
            let mut out = resampler.process(&[&left, &right], None).expect("resample");
            left_out.append(&mut out[0]);
            right_out.append(&mut out[1]);
            remaining -= next_frames;
            next_frames = resampler.input_frames_next();
        }

        if remaining > 0 {
            let mut out = resampler
                .process_partial(Some(&[&left, &right]), None)
                .expect("resample");
            left_out.append(&mut out[0]);
            right_out.append(&mut out[1]);
        }

        let signal = left_out
            .into_iter()
            .zip(right_out.into_iter())
            .map(|(l, r)| [l, r])
            .collect();

        Sample {
            signal,
            state: None,
        }
    }

    fn reverse(&self) -> Self {
        let mut out = self.signal.clone();
        out.reverse();
        Sample {
            signal: out,
            state: None,
        }
    }

    fn split(&self, at: usize) -> Vec<Self> {
        let (left, right) = self.signal.split_at(at);
        vec![
            Sample {
                signal: left.to_vec(),
                state: None,
            },
            Sample {
                signal: right.to_vec(),
                state: None,
            },
        ]
    }

    fn pitch(&self, cents: i32) -> Self {
        self.resample(HZ, freq_pitch(cents))
    }

    fn amplitude(&self, ratio: f32) -> Self {
        let signal = self
            .signal
            .par_iter()
            .map(|[l, r]| [l * ratio, r * ratio])
            .collect();
        Sample {
            signal,
            state: None,
        }
    }

    fn transforms(&self, n: usize) -> Vec<Self> {
        use rand::{
            distributions::{Distribution, Uniform},
            thread_rng,
        };
        let type_dist = Uniform::new(0, 4);
        let split_dist = Uniform::new(self.signal.len() / 4, 3 * self.signal.len() / 4);
        let pitch_dist = Uniform::new(-200, 200);
        let amplitude_dist = Uniform::new(0.5, 2.);
        let mut rng = thread_rng();
        (0..n)
            .flat_map(|_| match type_dist.sample(&mut rng) {
                0 => vec![self.reverse()],
                1 => self.split(split_dist.sample(&mut rng)),
                2 => vec![self.pitch(pitch_dist.sample(&mut rng))],
                3 => vec![self.amplitude(amplitude_dist.sample(&mut rng))],
                _ => unreachable!(),
            })
            .collect()
    }

    fn xcorr(
        &mut self,
        planner: &Mutex<FftPlanner<f32>>,
        (target_fft_left, target_fft_right): (&[Complex32], &[Complex32]),
    ) {
        // xcorr = ifft(fft(target) * conj(fft(self)))

        if self.state.is_some() {
            return;
        }

        assert!(target_fft_left.len() >= self.signal.len());
        // prepare self signal; pad to target length, and convert to complex
        let (mut left, mut right): (Vec<_>, Vec<_>) = self
            .signal
            .iter()
            .map(|&[l, r]| (Complex32::new(l, 0.), Complex32::new(r, 0.)))
            .chain(
                (self.signal.len()..target_fft_left.len())
                    .map(|_| (Complex32::new(0., 0.), Complex32::new(0., 0.))),
            )
            .unzip();

        // fft(self)
        let fft = planner
            .lock()
            .expect("lock fft planner")
            .plan_fft_forward(target_fft_left.len());
        fft.process(&mut left);
        fft.process(&mut right);

        // conj([fft(self)]) * [fft(target)]
        for i in 0..left.len() {
            left[i] = left[i].conj() * target_fft_left[i];
        }
        for i in 0..right.len() {
            right[i] = right[i].conj() * target_fft_right[i];
        }

        // ifft([conj(fft(self)) * fft(target)])
        let ifft = planner
            .lock()
            .expect("lock fft planner")
            .plan_fft_inverse(target_fft_left.len());
        ifft.process(&mut left);
        ifft.process(&mut right);

        // find delay and value (signal dot product) of peak
        let (l_delay, l_dot) = left
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.re.partial_cmp(&b.re).unwrap())
            .unwrap();
        let (r_delay, r_dot) = right
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.re.partial_cmp(&b.re).unwrap())
            .unwrap();

        self.state = Some((((l_delay + r_delay) / 2) as _, (l_dot.re + r_dot.re) / 2.));
    }
}

fn main() {
    let mut planner = FftPlanner::new();

    let target = Sample::from_wav("target.wav");
    let (mut target_fft_left, mut target_fft_right): (Vec<_>, Vec<_>) = target
        .signal
        .iter()
        .map(|&[l, r]| (Complex32::new(l, 0.), Complex32::new(r, 0.)))
        .chain(
            (target.signal.len()..target.signal.len().next_power_of_two())
                .map(|_| (Complex32::new(0., 0.), Complex32::new(0., 0.))),
        )
        .unzip();

    let fft = planner.plan_fft_forward(target_fft_left.len());
    fft.process(&mut target_fft_left);
    fft.process(&mut target_fft_right);

    let planner = Mutex::new(planner);

    let mut all = read_dir("samples")
        .expect("read samples folder")
        .filter_map(|file| file.ok())
        .filter(|file| {
            file.file_type().unwrap().is_file() && file.path().extension().unwrap() == "wav"
        })
        .par_bridge()
        .flat_map(|file| Sample::from_wav(file.path()).chunked(secs_to_frames(SAMPLE_CHUNK_SECS)))
        .collect::<Vec<_>>();

    let mut out = Sample::new(target.signal.len());

    for i in 0..CYCLES {
        for _ in 0..GENERATIONS_PER_CYCLE {
            all.append(
                &mut all
                    .par_iter()
                    .flat_map(|s| s.transforms(OFFSPRINGS_PER_SAMPLE))
                    .collect(),
            );
        }

        println!("Cycle {i} ({} samples)", all.len());

        all.par_iter_mut()
            .for_each(|s| s.xcorr(&planner, (&target_fft_left, &target_fft_right)));
        all.par_sort_unstable_by(|a, b| {
            a.state.unwrap().1.partial_cmp(&b.state.unwrap().1).unwrap()
        });

        let best = &all[0];
        let best_delay = best.state.unwrap().0 as usize;
        out.signal[best_delay..best_delay + best.signal.len()].copy_from_slice(&best.signal);

        all.drain((all.len() / 2).min(SURVIVAL_CAPACITY)..);
    }

    out.save_to_wav("out.wav");
}
