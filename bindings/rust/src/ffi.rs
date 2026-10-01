use std::os::raw::c_void;
unsafe extern "C"{
    pub fn tensor_create(shape:*const usize, shape_len:usize)->*mut c_void;
    pub fn tensor_free(t: *mut c_void);
    pub fn tensor_set_data(t: *mut c_void, data: *const f32, len:usize);
    pub fn tensor_get_data(t: *mut c_void, out_buffer:*mut f32, len:usize);
    pub fn tensor_ndim(t: *mut c_void)->usize;
    pub fn tensor_get_shape(t: *mut c_void, out_shape: *mut usize, out_len:usize); 
    pub fn tensor_set_elem(t: *mut c_void, idx: *const usize, idx_len:usize, value:f32);
    pub fn tensor_size(t: *mut c_void)->usize;
    pub fn tensor_matmul(a: *mut c_void , b: *mut c_void)->*mut c_void;
    pub fn tensor_addition(a: *mut c_void , b: *mut c_void)->*mut c_void;

}