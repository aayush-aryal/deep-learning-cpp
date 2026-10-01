mod ffi;
use ffi::*;


mod tensor;
use tensor::Tensor;




fn main(){
    // let shape=[1usize,2,2];
    // let shape_b=[2usize];
    // let shape_a=[1usize,2usize,2usize];
    // unsafe{
    //     let handle=tensor_create(shape.as_ptr(),shape.len());
    //     println!("Got handle: {:?}", handle);
    //     tensor_free(handle);
    //     println!("Done, without crashing! Yippee!!");
    //     let tensor_a= tensor_create(shape_a.as_ptr(), shape_a.len());
    //     let tensor_b= tensor_create(shape_b.as_ptr(),shape_b.len());
    //     let idx_a= [0usize,1,1];
    //     tensor_set_elem(tensor_a,idx_a.as_ptr(),idx_a.len(),1.0);
    //     let idx_b= [1usize];
    //     tensor_set_elem(tensor_b,idx_b.as_ptr(),idx_b.len(),1.0);
    //     let res= tensor_addition(tensor_a,tensor_b); 
    //     let res_size = tensor_size(res);
    //     let mut out_buffer = vec![0.0f32; res_size];
    //     tensor_get_data(res, out_buffer.as_mut_ptr(), out_buffer.len());
    //     println!("Result: {:?}", out_buffer);
    // }

    let mut a=Tensor::new(&[1,2,2]);
    a.set(&[0,1,1],1.0);

    let mut b= Tensor::new(&[1]);
    b.set(&[0],1.0);

    let c= a.add(&b);

    println!("{}",a);
    println!("{}",b);
    println!("{}",c);

}