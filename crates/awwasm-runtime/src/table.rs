//! Table instance implementation.
//!
//! A table instance is the runtime representation of a table,
//! holding references (currently just function references).

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::values::AwwasmFuncAddr;
use crate::error::AwwasmTrap;

/// Table type - describes the limits and element type of a table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AwwasmTableType {
    /// Minimum number of elements.
    pub min: u32,
    /// Maximum number of elements (if specified).
    pub max: Option<u32>,
    /// Element type (currently only funcref supported).
    pub elem_type: AwwasmElemType,
}

/// Element type for tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AwwasmElemType {
    /// Function reference.
    FuncRef,
    /// External reference (future).
    ExternRef,
}

impl AwwasmTableType {
    /// Create a new table type for function references.
    pub fn funcref(min: u32, max: Option<u32>) -> Self {
        Self {
            min,
            max,
            elem_type: AwwasmElemType::FuncRef,
        }
    }
}

/// Table instance - runtime representation of a table.
///
/// Tables hold references. Currently we only support funcref,
/// represented as Option<AwwasmFuncAddr> where None is null.
#[derive(Debug, Clone)]
pub struct AwwasmTableInst {
    /// The table type (limits + element type).
    pub type_: AwwasmTableType,
    /// The table elements (None = null reference).
    pub elem: Vec<Option<AwwasmFuncAddr>>,
}

impl AwwasmTableInst {
    /// Create a new table instance with the given type.
    ///
    /// Initializes all elements to null.
    pub fn new(type_: AwwasmTableType) -> Self {
        let size = type_.min as usize;
        Self {
            type_,
            elem: vec![None; size],
        }
    }

    /// Get the current size.
    #[inline]
    pub fn size(&self) -> u32 {
        self.elem.len() as u32
    }

    /// Get an element at the given index.
    #[inline]
    pub fn get(&self, index: u32) -> Result<Option<AwwasmFuncAddr>, AwwasmTrap> {
        let idx = index as usize;
        if idx >= self.elem.len() {
            return Err(AwwasmTrap::TableOutOfBounds {
                index,
                table_size: self.elem.len() as u32,
            });
        }
        Ok(self.elem[idx])
    }

    /// Set an element at the given index.
    #[inline]
    pub fn set(&mut self, index: u32, value: Option<AwwasmFuncAddr>) -> Result<(), AwwasmTrap> {
        let idx = index as usize;
        if idx >= self.elem.len() {
            return Err(AwwasmTrap::TableOutOfBounds {
                index,
                table_size: self.elem.len() as u32,
            });
        }
        self.elem[idx] = value;
        Ok(())
    }

    /// Grow the table by the given number of elements.
    ///
    /// Returns the previous size on success, or None if growth
    /// would exceed the maximum.
    pub fn grow(&mut self, delta: u32, init: Option<AwwasmFuncAddr>) -> Option<u32> {
        let old_size = self.size();
        let new_size = old_size.checked_add(delta)?;

        // Check against maximum
        if let Some(max) = self.type_.max {
            if new_size > max {
                return None;
            }
        }

        // Extend with the init value
        self.elem.resize(new_size as usize, init);
        Some(old_size)
    }

    /// Fill a range of elements with a value.
    pub fn fill(&mut self, offset: u32, value: Option<AwwasmFuncAddr>, count: u32) -> Result<(), AwwasmTrap> {
        let start = offset as usize;
        let end = start.checked_add(count as usize).ok_or(AwwasmTrap::TableOutOfBounds {
            index: offset,
            table_size: self.elem.len() as u32,
        })?;

        if end > self.elem.len() {
            return Err(AwwasmTrap::TableOutOfBounds {
                index: offset + count - 1,
                table_size: self.elem.len() as u32,
            });
        }

        self.elem[start..end].fill(value);
        Ok(())
    }

    /// Copy elements within the table.
    pub fn copy_within(&mut self, dst: u32, src: u32, count: u32) -> Result<(), AwwasmTrap> {
        let table_size = self.elem.len() as u32;

        // Check source bounds
        if src.checked_add(count).map_or(true, |end| end > table_size) {
            return Err(AwwasmTrap::TableOutOfBounds {
                index: src,
                table_size,
            });
        }

        // Check destination bounds
        if dst.checked_add(count).map_or(true, |end| end > table_size) {
            return Err(AwwasmTrap::TableOutOfBounds {
                index: dst,
                table_size,
            });
        }

        // Use rotate to handle overlapping copies correctly
        let src_range = src as usize..(src + count) as usize;
        let dst_start = dst as usize;
        
        // Clone the source slice first if there's overlap
        if (src..src + count).contains(&dst) || (dst..dst + count).contains(&src) {
            let src_data: Vec<_> = self.elem[src_range].to_vec();
            self.elem[dst_start..dst_start + count as usize].copy_from_slice(&src_data);
        } else {
            self.elem.copy_within(src_range, dst_start);
        }

        Ok(())
    }
}
