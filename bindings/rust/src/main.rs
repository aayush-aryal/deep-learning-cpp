use std::os::raw::c_void;

unsafe extern "C"{
    fn tensor_create(shape:*const usize, shape_len:usize)->*mut c_void;
    fn tensor_free(t: *mut c_void);
}

fn main(){
    let shape=[1usize,2,2];
    unsafe{
        let handle=tensor_create(shape.as_ptr(),shape.len());
        println!("Got handle: {:?}", handle);
        tensor_free(handle);
        println!("Done, without crashing! Yippee!!")
    }

}