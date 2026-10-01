use crate::ffi::*;

use std::os::raw::c_void;

pub struct Tensor{
    ptr: *mut c_void,
    len:usize,
    shape:Vec<usize>,
}

// main tensor implementation
impl Tensor{

    // constructor
    pub fn new(shape:&[usize])->Self{
        let len= shape.iter().product();
        let ptr= unsafe{tensor_create(shape.as_ptr(),shape.len())};
        Tensor{ptr,len, shape:shape.to_vec()}
    }

    fn from_ptr(ptr:*mut c_void)->Self{
        let len=unsafe{tensor_size(ptr)};
        let ndim=unsafe{tensor_ndim(ptr)};
        let mut shape=vec![0usize;ndim];
        unsafe{tensor_get_shape(ptr,shape.as_mut_ptr(),ndim)};
        Tensor{ptr,len,shape}
    }

    // set an index value of a tensor to a particular value
    pub fn set(&mut self, idx: &[usize], data:f32){
        unsafe{
            tensor_set_elem(self.ptr,idx.as_ptr(),idx.len(),data);
        }
    }

    pub fn get_data(&self)->Vec<f32>{
        let out_len= self.len;
        let mut out_buffer=vec![0.0f32;out_len];
        unsafe {
            tensor_get_data(self.ptr,out_buffer.as_mut_ptr(),out_len);
        }
        out_buffer
    }

    // pub fn get(&self, idx:&[usize])->f32{
    //     unsafe{
    //         tensor_get
    //     }

    // }

    pub fn print_shape(&self){
        print!("Shape: [");
        for i in 0..self.shape.len(){
            print!("{}",self.shape[i]);
            if (i!=self.shape.len()-1){
                print!(", ");

            }
        }
        print!("]\n");
    }

    fn print_tensor_recursive(&self,data:&[f32],shape:&[usize]){
        print!("[");
        if shape.len()==1{
            for i in 0..data.len(){
               print!("{}",data[i]);
                if (i!=data.len()-1){
                print!(", ");
                }
            }
        }else{
            let total_elems:usize= shape[1..].iter().product();
            for i in 0..shape[0]{
                let start=i*total_elems;
                let end= start+total_elems;
                self.print_tensor_recursive(&data[start..end], &shape[1..]);
                if i!=shape[0]-1 {print!(", ");}
            }
        }
        print!("]");
    }


    pub fn add(&self, other:&Tensor)->Tensor{
        let ptr= unsafe{tensor_addition(self.ptr,other.ptr)};
        Tensor::from_ptr(ptr)
    }

    pub fn matmul(&self, other:&Tensor)->Tensor{
        let ptr= unsafe{tensor_matmul(self.ptr,other.ptr)};
        Tensor::from_ptr(ptr)
    }
}

impl Drop for Tensor{
    fn drop(&mut self){
        unsafe{tensor_free(self.ptr)};
    }
}

impl std::fmt::Display for Tensor{
    fn fmt(&self, f:&mut std::fmt::Formatter)->std::fmt::Result{
        self.print_shape();
        print!("Data: ");
        let data = self.get_data();
        self.print_tensor_recursive(&data, &self.shape);
        println!();
        Ok(())
    }
}