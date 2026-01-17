//! Memory instance implementation.
//!
//! A memory instance is the runtime representation of a linear memory.
//! It holds a vector of bytes with page-granular sizing.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::error::AwwasmTrap;

/// WebAssembly page size in bytes (64 KiB).
pub const PAGE_SIZE: usize = 65536;

/// Memory type - describes the limits of a memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AwwasmMemoryType {
    /// Minimum number of pages.
    pub min: u32,
    /// Maximum number of pages (if specified).
    pub max: Option<u32>,
}

impl AwwasmMemoryType {
    /// Create a new memory type.
    pub fn new(min: u32, max: Option<u32>) -> Self {
        Self { min, max }
    }
}

/// Memory instance - runtime representation of linear memory.
///
/// The data vector always has a size that is a multiple of PAGE_SIZE.
#[derive(Debug, Clone)]
pub struct AwwasmMemInst {
    /// The memory type (limits).
    pub type_: AwwasmMemoryType,
    /// The raw bytes of memory.
    pub data: Vec<u8>,
}

impl AwwasmMemInst {
    /// Create a new memory instance with the given type.
    ///
    /// Allocates `min` pages of zeroed memory.
    pub fn new(type_: AwwasmMemoryType) -> Self {
        let size = (type_.min as usize) * PAGE_SIZE;
        Self {
            type_,
            data: vec![0u8; size],
        }
    }

    /// Get the current size in pages.
    #[inline]
    pub fn size_pages(&self) -> u32 {
        (self.data.len() / PAGE_SIZE) as u32
    }

    /// Get the current size in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Grow the memory by the given number of pages.
    ///
    /// Returns the previous size in pages on success, or None if growth
    /// would exceed the maximum or implementation limits.
    pub fn grow(&mut self, delta: u32) -> Option<u32> {
        let old_pages = self.size_pages();
        let new_pages = old_pages.checked_add(delta)?;

        // Check against maximum
        if let Some(max) = self.type_.max {
            if new_pages > max {
                return None;
            }
        }

        // Check against implementation limit (spec allows up to 2^16 pages for 32-bit)
        if new_pages > 65536 {
            return None;
        }

        // Extend with zeros
        let new_size = (new_pages as usize) * PAGE_SIZE;
        self.data.resize(new_size, 0);

        Some(old_pages)
    }

    /// Read bytes from memory.
    ///
    /// Returns a Trap if the access is out of bounds.
    pub fn read(&self, offset: u32, size: u32) -> Result<&[u8], AwwasmTrap> {
        let start = offset as usize;
        let end = start.checked_add(size as usize).ok_or(AwwasmTrap::MemoryOutOfBounds {
            offset,
            size,
            memory_size: self.data.len() as u32,
        })?;

        if end > self.data.len() {
            return Err(AwwasmTrap::MemoryOutOfBounds {
                offset,
                size,
                memory_size: self.data.len() as u32,
            });
        }

        Ok(&self.data[start..end])
    }

    /// Write bytes to memory.
    ///
    /// Returns a AwwasmTrap if the access is out of bounds.
    pub fn write(&mut self, offset: u32, data: &[u8]) -> Result<(), AwwasmTrap> {
        let start = offset as usize;
        let end = start.checked_add(data.len()).ok_or(AwwasmTrap::MemoryOutOfBounds {
            offset,
            size: data.len() as u32,
            memory_size: self.data.len() as u32,
        })?;

        if end > self.data.len() {
            return Err(AwwasmTrap::MemoryOutOfBounds {
                offset,
                size: data.len() as u32,
                memory_size: self.data.len() as u32,
            });
        }

        self.data[start..end].copy_from_slice(data);
        Ok(())
    }

    /// Read a single byte from memory.
    #[inline]
    pub fn read_u8(&self, offset: u32) -> Result<u8, AwwasmTrap> {
        let bytes = self.read(offset, 1)?;
        Ok(bytes[0])
    }

    /// Write a single byte to memory.
    #[inline]
    pub fn write_u8(&mut self, offset: u32, value: u8) -> Result<(), AwwasmTrap> {
        self.write(offset, &[value])
    }

    /// Read an i32 from memory (little-endian).
    #[inline]
    pub fn read_i32(&self, offset: u32) -> Result<i32, AwwasmTrap> {
        let bytes = self.read(offset, 4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Write an i32 to memory (little-endian).
    #[inline]
    pub fn write_i32(&mut self, offset: u32, value: i32) -> Result<(), AwwasmTrap> {
        self.write(offset, &value.to_le_bytes())
    }

    /// Read an i64 from memory (little-endian).
    #[inline]
    pub fn read_i64(&self, offset: u32) -> Result<i64, AwwasmTrap> {
        let bytes = self.read(offset, 8)?;
        Ok(i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    /// Write an i64 to memory (little-endian).
    #[inline]
    pub fn write_i64(&mut self, offset: u32, value: i64) -> Result<(), AwwasmTrap> {
        self.write(offset, &value.to_le_bytes())
    }

    /// Read an f32 from memory (little-endian).
    #[inline]
    pub fn read_f32(&self, offset: u32) -> Result<f32, AwwasmTrap> {
        let bytes = self.read(offset, 4)?;
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Write an f32 to memory (little-endian).
    #[inline]
    pub fn write_f32(&mut self, offset: u32, value: f32) -> Result<(), AwwasmTrap> {
        self.write(offset, &value.to_le_bytes())
    }

    /// Read an f64 from memory (little-endian).
    #[inline]
    pub fn read_f64(&self, offset: u32) -> Result<f64, AwwasmTrap> {
        let bytes = self.read(offset, 8)?;
        Ok(f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    /// Write an f64 to memory (little-endian).
    #[inline]
    pub fn write_f64(&mut self, offset: u32, value: f64) -> Result<(), AwwasmTrap> {
        self.write(offset, &value.to_le_bytes())
    }

    /// Fill a region of memory with a value.
    pub fn fill(&mut self, offset: u32, value: u8, size: u32) -> Result<(), AwwasmTrap> {
        let start = offset as usize;
        let end = start.checked_add(size as usize).ok_or(AwwasmTrap::MemoryOutOfBounds {
            offset,
            size,
            memory_size: self.data.len() as u32,
        })?;

        if end > self.data.len() {
            return Err(AwwasmTrap::MemoryOutOfBounds {
                offset,
                size,
                memory_size: self.data.len() as u32,
            });
        }

        self.data[start..end].fill(value);
        Ok(())
    }

    /// Copy a region within memory.
    pub fn copy_within(&mut self, dst: u32, src: u32, size: u32) -> Result<(), AwwasmTrap> {
        let mem_size = self.data.len() as u32;
        
        // Check source bounds
        if src.checked_add(size).map_or(true, |end| end > mem_size) {
            return Err(AwwasmTrap::MemoryOutOfBounds {
                offset: src,
                size,
                memory_size: mem_size,
            });
        }

        // Check destination bounds
        if dst.checked_add(size).map_or(true, |end| end > mem_size) {
            return Err(AwwasmTrap::MemoryOutOfBounds {
                offset: dst,
                size,
                memory_size: mem_size,
            });
        }

        self.data.copy_within(src as usize..(src + size) as usize, dst as usize);
        Ok(())
    }
}
