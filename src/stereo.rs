use itertools::Itertools;
use rubato::{FftFixedInOut, Resampler};
use std::{fs::File, path::Path};

pub type Stereo = [Vec<f32>; 2];
pub type StereoRef<'a> = [&'a [f32]; 2];

pub fn freq_pitch(hz: u32, cents: i32) -> u32 {
    (2.0f32.powf(cents as f32 / 1200.0) * hz as f32) as _
}

pub fn stereo_to_owned(input: StereoRef) -> Stereo {
    [input[0].to_vec(), input[1].to_vec()]
}

pub fn stereo_borrow(input: &Stereo) -> StereoRef {
    [&input[0], &input[1]]
}

pub fn spectro(input: StereoRef, hz: u32, xbins: usize, ybins: usize) -> Vec<f32> {
    assert!(ybins.is_power_of_two());
    sonogram::SpecOptionsBuilder::new(ybins)
        .load_data_from_memory_f32(input[0].to_vec(), hz)
        .downsample(3)
        .normalise()
        .set_window_fn(sonogram::hann_function)
        .build()
        .expect("spectrogram")
        .compute()
        .to_buffer(sonogram::FrequencyScale::Log, xbins, ybins)
}

pub fn normalize(input: StereoRef) -> Stereo {
    let mut out = stereo_to_owned(input);
    let l_max = out[0]
        .iter()
        .max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap())
        .copied()
        .unwrap_or(0.);
    let r_max = out[1]
        .iter()
        .max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap())
        .copied()
        .unwrap_or(0.);
    if l_max > 0. {
        out[0].iter_mut().for_each(|x| *x /= l_max);
    }
    if r_max > 0. {
        out[1].iter_mut().for_each(|x| *x /= r_max);
    }
    out
}

pub fn rms(input: StereoRef) -> f32 {
    let mut sum = 0.;
    sum += input[0].iter().map(|x| x * x).sum::<f32>();
    sum += input[1].iter().map(|x| x * x).sum::<f32>();
    sum /= (input[0].len() * 2) as f32;
    sum.sqrt()
}

pub fn load_wav(file: impl AsRef<Path>, hz: u32) -> Stereo {
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

    let (l, r): (Vec<f32>, Vec<f32>) = interleaved.chunks(2).map(|x| (x[0], x[1])).unzip();
    resample(stereo_borrow(&[l, r]), header.sampling_rate, hz)
}

pub fn save_to_wav(input: StereoRef, file: impl AsRef<Path>, hz: u32) {
    let mut file = File::create(file).expect("write file");
    wav::write(
        wav::Header::new(wav::WAV_FORMAT_PCM, 2, hz, 16),
        &wav::BitDepth::Sixteen(
            input[0]
                .iter()
                .interleave(input[1].iter())
                .map(|x| (x.clamp(-1., 1.) * 0x8000 as f32) as i16)
                .collect(),
        ),
        &mut file,
    )
    .expect("encode wav");
}

pub fn reverse(input: StereoRef) -> Stereo {
    let mut output = stereo_to_owned(input);
    output[0].reverse();
    output[1].reverse();
    output
}

pub fn resample(input: StereoRef, f1: u32, f2: u32) -> Stereo {
    const CHUNK_SIZE: usize = 2048;

    if f1 == f2 {
        return stereo_to_owned(input);
    }

    let mut resampler =
        FftFixedInOut::new(f1 as _, f2 as _, CHUNK_SIZE, 2).expect("init resampler");
    let mut left_out = vec![];
    left_out.reserve(resampler.output_frames_max());
    let mut right_out = vec![];
    right_out.reserve(resampler.output_frames_max());

    let mut remaining = input[0].len();
    let mut next_frames = resampler.input_frames_next();
    while remaining > next_frames {
        let mut out = resampler
            .process(
                &[
                    &input[0][input[0].len() - remaining..],
                    &input[1][input[1].len() - remaining..],
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
                    &input[0][input[0].len() - remaining..],
                    &input[1][input[1].len() - remaining..],
                ]),
                None,
            )
            .expect("resample");
        left_out.append(&mut out[0]);
        right_out.append(&mut out[1]);
    }

    [left_out, right_out]
}

pub fn pitch(input: StereoRef, hz: u32, cents: i32) -> Stereo {
    resample(input, hz, freq_pitch(hz, -cents))
}

pub fn gain(input: StereoRef, ratio: f32) -> Stereo {
    let mut output = stereo_to_owned(input);
    output[0].iter_mut().for_each(|x| *x *= ratio);
    output[1].iter_mut().for_each(|x| *x *= ratio);
    output
}

pub fn add(a: StereoRef, b: StereoRef, delay: usize, mix: Option<f32>, trunc: bool) -> Stereo {
    if a[0].len() >= b[0].len() || trunc {
        let mut output = stereo_to_owned(a);
        let mix_a = mix.map(|x| 1. - x).unwrap_or(1.);
        let mix_b = mix.unwrap_or(1.);
        output[0].iter_mut().enumerate().for_each(|(i, x)| {
            if i >= delay && i < delay + b[0].len() {
                *x = mix_a * *x + mix_b * b[0][i - delay];
            }
        });
        output[1].iter_mut().enumerate().for_each(|(i, x)| {
            if i >= delay && i < delay + b[1].len() {
                *x = mix_a * *x + mix_b * b[1][i - delay];
            }
        });
        output
    } else {
        add(b, a, 0, mix.map(|x| 1. - x), true)
    }
}

pub fn append(a: StereoRef, b: StereoRef) -> Stereo {
    let mut output = [a[0].to_vec(), a[1].to_vec()];
    output[0].extend(b[0].iter());
    output[1].extend(b[1].iter());
    output
}

pub fn chunked(input: StereoRef, frames: usize) -> Vec<StereoRef> {
    let count = (input[0].len() + frames - 1) / frames;
    (0..count)
        .map(|i| {
            let delay = i * frames;
            let next = ((i + 1) * frames).min(input[0].len());
            [&input[0][delay..next], &input[1][delay..next]]
        })
        .collect()
}

pub fn split(input: StereoRef, at: usize) -> [StereoRef; 2] {
    let ((a_left, b_left), (a_right, b_right)) = (input[0].split_at(at), input[1].split_at(at));
    [[a_left, a_right], [b_left, b_right]]
}

pub fn shelf(input: StereoRef, low: bool, hz: u32, cutoff: u32, gain: f32) -> Stereo {
    let coeffs = biquad::Coefficients::<f32>::from_params(
        if low {
            biquad::Type::LowShelf(gain)
        } else {
            biquad::Type::HighShelf(gain)
        },
        biquad::Hertz::<f32>::from_hz(hz as f32).unwrap(),
        biquad::Hertz::<f32>::from_hz(cutoff as f32).unwrap(),
        100.,
    )
    .expect("biquad coeffs");

    let mut out = [vec![0.; input[0].len()], vec![0.; input[0].len()]];

    let mut filter = biquad::DirectForm1::<f32>::new(coeffs);
    for (i, x) in input[0].iter().copied().enumerate() {
        out[0][i] = biquad::Biquad::run(&mut filter, x);
    }

    biquad::Biquad::reset_state(&mut filter);
    for (i, x) in input[1].iter().copied().enumerate() {
        out[1][i] = biquad::Biquad::run(&mut filter, x);
    }

    out
}
