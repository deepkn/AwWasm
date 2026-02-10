//! Conversion utilities: parser types → runtime types.

use awwasm_parser::components::types::AwwasmMemoryParams;
use awwasm_parser::components::instructions::eval_const_init_expr;

use crate::error::AwwasmInstantiationError;
use crate::values::AwwasmValueType;
use crate::memory::AwwasmMemoryType;

/// Convert parser's `ParamType` to runtime's `AwwasmValueType`.
pub fn param_type_to_value_type(pt: &awwasm_parser::components::types::ParamType) -> Result<AwwasmValueType, AwwasmInstantiationError> {
    match pt {
        awwasm_parser::components::types::ParamType::I32 => Ok(AwwasmValueType::I32),
        awwasm_parser::components::types::ParamType::I64 => Ok(AwwasmValueType::I64),
        awwasm_parser::components::types::ParamType::IUnknown => Err(AwwasmInstantiationError::UnsupportedType {
            description: "unknown param type 0x00".into(),
        }),
    }
}

/// Convert parser's `AwwasmMemoryParams` to runtime's `AwwasmMemoryType`.
pub fn memory_params_to_type(params: &AwwasmMemoryParams) -> AwwasmMemoryType {
    AwwasmMemoryType::new(params.min, params.max)
}

/// Evaluate a constant initializer expression using the parser.
///
/// Delegates to `awwasm_parser::eval_const_init_expr` and converts
/// the result/error to runtime types.
pub fn eval_const_expr(code: &[u8]) -> Result<u32, AwwasmInstantiationError> {
    let value = eval_const_init_expr(code).map_err(|e| {
        AwwasmInstantiationError::InvalidConstExpr {
            description: e.to_string(),
        }
    })?;
    Ok(value as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use awwasm_parser::components::types::ParamType;

    #[test]
    fn test_eval_const_expr_i32() {
        assert_eq!(eval_const_expr(&[0x41, 0x00]).unwrap(), 0);
        assert_eq!(eval_const_expr(&[0x41, 0x2a]).unwrap(), 42);
        assert_eq!(eval_const_expr(&[0x41, 0x80, 0x01]).unwrap(), 128);
    }

    #[test]
    fn test_eval_const_expr_negative() {
        assert_eq!(eval_const_expr(&[0x41, 0x7f]).unwrap(), u32::MAX);
    }

    #[test]
    fn test_eval_const_expr_unsupported() {
        assert!(eval_const_expr(&[0x42, 0x00]).is_err());
    }

    #[test]
    fn test_param_type_conversion() {
        assert_eq!(param_type_to_value_type(&ParamType::I32).unwrap(), AwwasmValueType::I32);
        assert_eq!(param_type_to_value_type(&ParamType::I64).unwrap(), AwwasmValueType::I64);
        assert!(param_type_to_value_type(&ParamType::IUnknown).is_err());
    }

    #[test]
    fn test_memory_params_conversion() {
        let params = AwwasmMemoryParams { flags: 1, min: 1, max: Some(4) };
        let ty = memory_params_to_type(&params);
        assert_eq!(ty.min, 1);
        assert_eq!(ty.max, Some(4));
    }
}
