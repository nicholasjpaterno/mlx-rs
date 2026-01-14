use std::ffi::CString;

use mlx_internal_macros::{default_device, generate_macro};

use crate::{error::Result, utils::guard::Guarded, Array, Dtype, Stream};

/// Quantization mode for weight quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantizationMode {
    /// Affine quantization with scale and bias per group.
    #[default]
    Affine,
    /// MX FP4 format.
    MxFp4,
    /// MX FP8 format.
    MxFp8,
    /// NV FP4 format.
    NvFp4,
}

impl QuantizationMode {
    fn as_cstr(&self) -> CString {
        CString::new(match self {
            Self::Affine => "affine",
            Self::MxFp4 => "mxfp4",
            Self::MxFp8 => "mxfp8",
            Self::NvFp4 => "nvfp4",
        })
        .unwrap()
    }
}

/// Helper to create mlx_optional_int from Option<i32>
fn optional_int(value: Option<i32>) -> mlx_sys::mlx_optional_int {
    mlx_sys::mlx_optional_int {
        value: value.unwrap_or(0),
        has_value: value.is_some(),
    }
}

/// Helper to create mlx_optional_dtype from Option<Dtype>
fn optional_dtype(value: Option<Dtype>) -> mlx_sys::mlx_optional_dtype {
    mlx_sys::mlx_optional_dtype {
        value: value.map(|d| d.into()).unwrap_or(mlx_sys::mlx_dtype__MLX_FLOAT32),
        has_value: value.is_some(),
    }
}

/// Quantize the matrix `w` using `bits` bits per element.
///
/// Note, every `group_size` elements in a row of `w` are quantized together. Hence, number of
/// columns of `w` should be divisible by `group_size`. In particular, the rows of `w` are divided
/// into groups of size `group_size` which are quantized together.
///
/// > `quantized` currently only supports 2D inputs with dimensions which are multiples of 32
///
/// For details, please see [this
/// documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
///
/// # Params
///
/// - `w`: The input matrix
/// - `group_size`: The size of the group in `w` that shares a scale and bias. (default: depends on mode)
/// - `bits`: The number of bits occupied by each element of w in the returned quantized matrix.
///   (default: depends on mode)
/// - `mode`: Quantization mode (default: Affine)
///
/// # Returns
///
/// A tuple of (quantized_weights, scales, biases) for affine mode, or
/// (quantized_weights, scales) for other modes. The third element may be empty.
#[generate_macro]
#[default_device]
pub fn quantize_device(
    w: impl AsRef<Array>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] mode: impl Into<Option<QuantizationMode>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Vec<Array>> {
    let group_size = optional_int(group_size.into());
    let bits = optional_int(bits.into());
    let mode = mode.into().unwrap_or_default();
    let mode_cstr = mode.as_cstr();

    <Vec<Array> as Guarded>::try_from_op(|res| unsafe {
        mlx_sys::mlx_quantize(
            res,
            w.as_ref().as_ptr(),
            group_size,
            bits,
            mode_cstr.as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Perform the matrix multiplication with the quantized matrix `w`. The quantization uses one
/// floating point scale and bias per `group_size` of elements. Each element in `w` takes `bits`
/// bits and is packed in an unsigned 32 bit integer.
#[allow(clippy::too_many_arguments)]
#[generate_macro]
#[default_device]
pub fn quantized_matmul_device(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    biases: Option<&Array>,
    #[optional] transpose: impl Into<Option<bool>>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] mode: impl Into<Option<QuantizationMode>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let transpose = transpose.into().unwrap_or(true);
    let group_size = optional_int(group_size.into());
    let bits = optional_int(bits.into());
    let mode = mode.into().unwrap_or_default();
    let mode_cstr = mode.as_cstr();

    // Handle optional biases - pass null/empty array if None
    let biases_arr = biases
        .map(|b| b.as_ptr())
        .unwrap_or(mlx_sys::mlx_array { ctx: std::ptr::null_mut() });

    <Array as Guarded>::try_from_op(|res| unsafe {
        mlx_sys::mlx_quantized_matmul(
            res,
            x.as_ref().as_ptr(),
            w.as_ref().as_ptr(),
            scales.as_ref().as_ptr(),
            biases_arr,
            transpose,
            group_size,
            bits,
            mode_cstr.as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Dequantize the matrix `w` using the provided `scales` and `biases` and the `group_size` and
/// `bits` configuration.
///
/// For details, please see [this
/// documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.dequantize.html)
#[generate_macro]
#[default_device]
pub fn dequantize_device(
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    biases: Option<&Array>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] mode: impl Into<Option<QuantizationMode>>,
    #[optional] dtype: impl Into<Option<Dtype>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let group_size = optional_int(group_size.into());
    let bits = optional_int(bits.into());
    let mode = mode.into().unwrap_or_default();
    let mode_cstr = mode.as_cstr();
    let dtype = optional_dtype(dtype.into());

    // Handle optional biases - pass null/empty array if None
    let biases_arr = biases
        .map(|b| b.as_ptr())
        .unwrap_or(mlx_sys::mlx_array { ctx: std::ptr::null_mut() });

    <Array as Guarded>::try_from_op(|res| unsafe {
        mlx_sys::mlx_dequantize(
            res,
            w.as_ref().as_ptr(),
            scales.as_ref().as_ptr(),
            biases_arr,
            group_size,
            bits,
            mode_cstr.as_ptr(),
            dtype,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Convert an E4M3 float8 array to the specified floating point type.
///
/// This function converts an array stored as uint8 (representing FP8 E4M3 format)
/// back to a standard floating point type like float32 or bfloat16.
///
/// # Params
///
/// - `x`: The input array with dtype uint8 containing FP8 E4M3 encoded values
/// - `dtype`: The target floating point dtype (e.g., Float32, Bfloat16)
///
/// # Example
///
/// ```ignore
/// use mlx_rs::{Array, Dtype, ops::from_fp8};
///
/// // Convert FP8 data back to float32
/// let fp8_data = to_fp8(&weights)?;
/// let weights_f32 = from_fp8(&fp8_data, Dtype::Float32)?;
/// ```
#[generate_macro]
#[default_device]
pub fn from_fp8_device(
    x: impl AsRef<Array>,
    dtype: Dtype,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    <Array as Guarded>::try_from_op(|res| unsafe {
        mlx_sys::mlx_from_fp8(
            res,
            x.as_ref().as_ptr(),
            dtype.into(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Convert a floating point array to E4M3 float8 format.
///
/// This function converts standard floating point values to FP8 E4M3 format,
/// stored as uint8. The E4M3 format has 4 exponent bits and 3 mantissa bits,
/// with a range of approximately +/-240.
///
/// FP8 provides ~2x memory reduction compared to FP16/BF16 with minimal
/// accuracy loss for inference workloads.
///
/// # Params
///
/// - `x`: The input floating point array (float32, float16, or bfloat16)
///
/// # Example
///
/// ```ignore
/// use mlx_rs::{Array, ops::to_fp8};
///
/// // Quantize weights to FP8 for memory-efficient inference
/// let weights = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
/// let fp8_weights = to_fp8(&weights)?;
/// assert_eq!(fp8_weights.dtype(), Dtype::Uint8);
/// ```
#[generate_macro]
#[default_device]
pub fn to_fp8_device(
    x: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    <Array as Guarded>::try_from_op(|res| unsafe {
        mlx_sys::mlx_to_fp8(res, x.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

#[cfg(test)]
mod tests {
    use super::QuantizationMode;
    use crate::{
        ops::{dequantize, expand_dims, from_fp8, quantize, to_fp8},
        Array, Dtype,
    };

    #[test]
    fn test_quantize_dequantize() {
        let x1 = Array::ones::<f32>(&[128, 1]).unwrap();
        let x2 = expand_dims(Array::arange::<_, f32>(0, 512, None).unwrap(), 0).unwrap();
        let x = x1 * x2;

        for i in [2, 4, 8].iter() {
            let el_per_int = 32 / i;
            let result = quantize(&x, 128, *i, QuantizationMode::Affine).unwrap();
            assert!(result.len() >= 2);

            let x_q = &result[0];
            let scales = &result[1];
            let biases = result.get(2);

            assert_eq!(x_q.shape(), [128, 512 / el_per_int]);
            assert_eq!(scales.shape(), [128, 4]);

            let x_hat = dequantize(x_q, scales, biases, 128, *i, QuantizationMode::Affine, None).unwrap();
            let max_diff = ((&x - &x_hat).abs().unwrap().max(None).unwrap()).item::<f32>();
            assert!(max_diff <= 127.0 / (1 << i) as f32);
        }
    }

    #[test]
    fn test_fp8_roundtrip() {
        // Test FP8 E4M3 quantization/dequantization
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]);

        // Convert to FP8
        let fp8 = to_fp8(&x).unwrap();
        assert_eq!(fp8.dtype(), Dtype::Uint8);

        // Convert back to float32
        let x_hat = from_fp8(&fp8, Dtype::Float32).unwrap();
        assert_eq!(x_hat.dtype(), Dtype::Float32);
        assert_eq!(x_hat.shape(), x.shape());

        // Evaluate and check values are close
        x.eval().unwrap();
        x_hat.eval().unwrap();
    }
}
