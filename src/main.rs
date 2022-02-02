use std::fs::OpenOptions;
use rand::rngs::ThreadRng;
use std::process::Command;
use std::time::Instant;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};

use std::io::Write;
use std::io::BufWriter;
use std::fs;
use std::fs::File;
use chrono::{Utc};

use float_ord::FloatOrd;

use std::ops;
use rand::prelude::*;
use std::ops::Index;
// Utilities + Structs

struct RandSource 
{
    inner: ThreadRng
}

impl RandSource 
{
    fn new() -> RandSource 
    {   
        RandSource{ inner: thread_rng()}
    }

    fn in_unit_sphere(&mut self) -> Vector3
    {
        loop 
        {
            let to_return = Vector3::new(-1.0 + 2.0 * self.inner.gen::<f32>(), -1.0 + 2.0 * self.inner.gen::<f32>(), -1.0 + 2.0 * self.inner.gen::<f32>());
            if to_return.length_squared() <= 1.0 
            {
                return to_return;
            }
        }
    }

    fn on_unit_sphere(&mut self) -> Vector3 
    {
        self.in_unit_sphere().unit() 
    }
}

fn clamp(a: f32) -> f32
{
    if a > 1.0 
    {
        1.0
    } else if a < 0.0 {
        0.0
    } else {
        a
    }
}

fn lerp_f(t: f32, a : f32, b: f32) -> f32 
{
    (1.0 - t) * a + t * b
}

fn lerp_v(t: f32, a: Vector3, b: Vector3) -> Vector3 
{
    a * (1.0 - t) + b * t
}

#[derive(Copy, Clone, Debug)]
struct Vector3 
{
    inner: [f32; 3]
}

impl Vector3 
{    
    fn x(&self) -> f32
    {
        self.inner[0]
    }

    fn r(&self) -> f32
    {
        self.inner[0]
    }

    fn y(&self) -> f32
    {
        self.inner[1]
    }   
    
    fn g(&self) -> f32
    {
        self.inner[1]
    }

    fn z(&self) -> f32
    {
        self.inner[2]
    }

    fn b(&self) -> f32
    {
        self.inner[2]
    }
}

impl Vector3 
{

    fn BLACK() -> Vector3 
    {
        Vector3::new(0.0,0.0,0.0)
    }

    fn WHITE() -> Vector3 
    {
        Vector3::new(1.0,1.0,1.0)
    }

    fn new(x: f32, y: f32, z: f32) -> Vector3 
    {
        Vector3 
        {
            inner: [x, y, z]
        }
    }

    fn dot(self, other: Vector3) -> f32 
    {
        let mult = self * other;
        mult.x() + mult.y() + mult.z()
    }

    fn cross(self, other: Vector3) -> Vector3 
    {
        return Vector3
        {    
            inner: 
            [
                self.y() * other.z() - self.z() * other.y(),
                self.z() * other.x() - self.x() * other.z(),
                self.x() * other.y() - self.y() * other.x()
            ]
        }
    }

    fn unit(self) -> Vector3 
    {
        let l = self.length();
        self * (1.0/l)
    }

    fn length(&self) -> f32 
    {
        self.length_squared().sqrt()
    }

    fn length_squared(&self) -> f32 
    {
        self.x() * self.x() + self.y() * self.y() + self.z() * self.z()
    }

    fn reflect_by(&self, normal: &Vector3) -> Vector3
    {
        *self +- 2.0*self.dot(*normal) * *normal
    }
}

impl std::ops::Neg for Vector3 
{
    type Output = Self;

    fn neg(self) -> Self::Output 
    {
        Vector3{ inner: [-self.x(), -self.y(), -self.z()] }
    }
}

impl std::ops::Add for Vector3 
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output 
    {
        Vector3{ inner: [self.x()+other.x(), self.y()+other.y(), self.z()+other.z()] }
    }
}

impl std::ops::Mul<Vector3> for f32 
{
    type Output = Vector3;

    fn mul(self, other: Vector3) -> Self::Output
    {   
        Vector3{ inner: [self*other.x(), self*other.y(), self*other.z()] }
    }
}

impl std::ops::Mul<Vector3> for Vector3 
{
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output
    {   
        Vector3{ inner: [self.x()*other.x(), self.y()*other.y(), self.z()*other.z()] }
    }
}

impl std::ops::Mul<f32> for Vector3 
{
    type Output = Self;

    fn mul(self, scalar: f32) -> Self::Output
    {   
        Vector3{ inner: [self.x()*scalar, self.y()*scalar, self.z()*scalar] }
    }
}

impl std::ops::Div<f32> for Vector3 
{
    type Output = Self;

    fn div(self, scalar: f32) -> Self::Output
    {   
        return self * (1.0/scalar);
    }
}

struct Ray 
{
    position: Vector3,
    direction: Vector3
}

impl Ray 
{
    fn point_at(&self, scalar: f32) -> Vector3 
    {
        let d = self.direction * scalar;
        self.position + d
    }
}

#[derive(Clone, Copy)]
struct SizeAspect 
{
    w_h: [i32; 2],
    aspect_ratio: f32,
}

impl SizeAspect 
{   
    fn new(width_px: i32, aspect_ratio: f32) -> SizeAspect
    {
        SizeAspect 
        {
            w_h : [width_px, (width_px as f32 * (1.0 / aspect_ratio)) as i32],
            aspect_ratio : aspect_ratio,
        }
    }   
}

struct CameraInfo 
{
    size_aspect: SizeAspect,
 
    viewport_height : f32,
    viewport_width : f32,

    focal_length: f32,

    origin: Vector3,
    
    horizontal_vector : Vector3,
    vertical_vector : Vector3,
    lower_left_corner: Vector3,
}

impl CameraInfo 
{
    // fn new(width_px: i32, aspect_ratio : f32) -> CameraInfo
    // {
    //     let viewport_height = 2.0;
    //     let viewport_width = aspect_ratio * viewport_height;

    //     let focal_length = 1.0;

    //     let origin = Vector3 { inner: [0.0, 0.0, 0.0]};

    //     let horizontal_vector = Vector3 { inner: [viewport_width, 0.0 ,0.0]};
    //     let vertical_vector = Vector3 { inner: [0.0, viewport_height, 0.0]};
    //     let lower_left_corner = origin + -horizontal_vector/2.0 + -vertical_vector/2.0 +-Vector3 { inner: [0.0, 0.0, focal_length]};

    //     CameraInfo 
    //     {
    //         w_h : [width_px, (width_px as f32 / aspect_ratio) as i32 ],
    //         aspect_ratio,

    //         viewport_height,
    //         viewport_width,

    //         focal_length,

    //         origin,

    //         horizontal_vector,
    //         vertical_vector,
    //         lower_left_corner
    //     }
    // }

    fn new(lerp: [f32;2], size_aspect: SizeAspect) -> CameraInfo
    {
        let viewport_height = 2.0;
        let viewport_width = size_aspect.aspect_ratio * viewport_height;

        let focal_length = 1.0;

        let origin = Vector3 { inner: [0.0, 0.0, 0.0]};

        let horizontal_vector = Vector3 { inner: [viewport_width, 0.0 ,0.0]};
        let vertical_vector = Vector3 { inner: [0.0, viewport_height, 0.0]};
        let lower_left_corner = origin + -horizontal_vector/2.0 + -vertical_vector/2.0 +-Vector3 { inner: [0.0, 0.0, focal_length]};

        CameraInfo 
        {
            size_aspect,

            viewport_height,
            viewport_width,

            focal_length,

            origin,

            horizontal_vector,
            vertical_vector,
            lower_left_corner
        }
    }
}

#[derive(Serialize, Deserialize)]
struct RenderInfo 
{
    number: i32,
    date_time_string: String,

    w_h: [i32; 2],
    total_pixels : i64,

    total_time_ms: u128,
    
    ms_per_pixel : f64,
}

impl RenderInfo 
{
    fn new(number: i32, w_h: [i32; 2], total_time_ms: u128) -> RenderInfo 
    {
        RenderInfo 
        {
            number,
            date_time_string: Utc::now().format("%d_%m_%Y_%H_%M_%S").to_string(),

            w_h,
            total_pixels: (w_h[0] * w_h[1]) as i64,

            total_time_ms,

            ms_per_pixel : (total_time_ms as f64 / (w_h[0] * w_h[1]) as f64)
        }
    }
}

struct Warhol 
{
    sample_count: i32,
    w_h_dim : [i32; 2],
    px_fn : fn(i32, [f32; 2], [i32; 2], &SizeAspect, &HittableCollection, &mut RandSource) -> Vector3
}

fn main() 
{
    {
        let world = build_world();
        let mut rand = RandSource::new();

        {
            let warhol = Warhol { sample_count: 10, w_h_dim: [1,1], px_fn : color_px };
            let sa = SizeAspect::new(256, 16.0/9.0);

            open_image_viewer(render(&warhol, &sa, &world, &mut rand));
        }

        // {
        //     let warhol = Warhol { sample_count: 50, w_h_dim: [1,1], px_fn : color_px };
        //     let sa = SizeAspect::new(1024, 16.0/9.0);

        //     open_image_viewer(render(&warhol, &sa, &world, &mut rand));
        // }
    }
}

fn open_image_viewer(image_path: String)
{
    let viewer_path = "C:\\Program Files\\ImageGlass\\ImageGlass.exe";
    Command::new(viewer_path).arg(image_path).spawn().expect("could not launch viewer");
}

fn render(warhol: &Warhol, size_aspect: &SizeAspect, world: &HittableCollection, rand: &mut RandSource) -> String
{
    let timer = Instant::now();

    let master_w_h = [warhol.w_h_dim[0] * size_aspect.w_h[0], warhol.w_h_dim[1] * size_aspect.w_h[1]];

    let base_path = "X:\\Image_Testing";

    fn determine_index(base_path: &str) -> i32
    {
        let paths = fs::read_dir(base_path).unwrap();
    
        let max = 
            paths.
            flatten().
            map(|path| path.file_name().into_string()).
            flatten().
            map(|full_file_name| full_file_name.split(".").map(|s| s.to_string()).next()).
            flatten().
            map(|file_name_no_ext| file_name_no_ext.parse::<i32>()).
            flatten().
            max();
        
        match max 
        {
            Some(i) => i+1,
            None => 1
        }
    }

    let index = determine_index(base_path);

    let image_file_name : String = [index.to_string(), ".ppm".to_string()].into_iter().collect();
    let image_file_path : PathBuf = [base_path, &image_file_name].iter().collect();

    fn color_as_bytes(color: Vector3) -> Vec<u8> 
    {
        let s = format!("{} {} {}\n", (color.x() * 255.0) as i32, (color.y() * 255.0) as i32, (color.z() * 255.0) as i32);
        s.as_bytes().to_vec()
    } 

    let image_file = File::create(image_file_path.clone()).expect("Unable to create file");
    let mut image_buffer = BufWriter::new(image_file);

    image_buffer.write("P3\n".as_bytes()).expect("cannot_write"); // Triples
    image_buffer.write(format!("{} {}\n", master_w_h[0], master_w_h[1]).as_bytes()).expect("cannot_write"); // Width, Height
    image_buffer.write("255\n".as_bytes()).expect("cannot_write"); // 255 max
 
    fn color_warhol(warhol: &Warhol, size_aspect: &SizeAspect, actual_coordinate : [i32; 2], world: &HittableCollection, rand: &mut RandSource) -> Vector3 
    {
        fn lerp_value(coordinate: i32, size: i32, dim: i32) -> f32
        {   
            if dim == 1 { 0.0 } 
            else { ((coordinate/size) as f32) / (dim-1) as f32 }
        }
            
        let modulo_coordinate = [actual_coordinate[0] % size_aspect.w_h[0], actual_coordinate[1] % size_aspect.w_h[1]];
    
        let lerp = [lerp_value(actual_coordinate[0], size_aspect.w_h[0], warhol.w_h_dim[0]),
                    lerp_value(actual_coordinate[1], size_aspect.w_h[1], warhol.w_h_dim[1])];
    
        return (warhol.px_fn)(warhol.sample_count, lerp, modulo_coordinate, size_aspect, world, rand);
    }

    for h in (0..master_w_h[1]).rev() 
    {
        for w in 0..master_w_h[0] 
        {
            let c = color_warhol(warhol, size_aspect, [w,h], world, rand);
            image_buffer.write(&color_as_bytes(c)).expect("cannot_write");
        } 
    }

    image_buffer.flush().expect("Could not flush");

    let timer_elapsed = timer.elapsed().as_millis();

    let render_info = RenderInfo::new(index, size_aspect.w_h.clone(), timer_elapsed);

    fn render_info_as_bytes(r: RenderInfo) -> Vec<u8>
    {
        serde_json::to_string(&r).unwrap().as_bytes().to_vec()
    }

    let info_file_name : String = [index.to_string(), ".json".to_string()].into_iter().collect();
    let info_file_path : PathBuf = [base_path, &info_file_name].iter().collect();
    
    let info_file = File::create(info_file_path).expect("Unable to create file");
    let mut info_buffer = BufWriter::new(info_file);

    info_buffer.write_all(&render_info_as_bytes(render_info)).expect("cannot_write");
    info_buffer.flush().expect("Could not flush");

    return image_file_path.to_str().unwrap().to_string();
}

#[derive(Copy, Clone)]
struct HitRecord
{
    distance : f32,
    hit_point: Vector3,
    normal_unit :Vector3,
    front_face: bool,

    material: Material
}

impl HitRecord
{
    fn set_front(&mut self) 
    {
        if !self.front_face 
        {
            self.normal_unit = -self.normal_unit;
            self.front_face = true;
        }
    }
}

trait Hittable 
{
    fn hit(&self, min_max: [f32; 2], r: &Ray) -> Option<HitRecord>; 
}

struct TrivialHitter
{

}

impl Hittable for TrivialHitter 
{
    fn hit(&self, _bounds: [f32; 2], _r: &Ray) -> Option<HitRecord> { return Some(HitRecord { distance: 0.0, hit_point: Vector3::new(0.0,0.0,0.0), normal_unit: Vector3::new(1.0,0.0,0.0), front_face: true, material: Material::default() }) }
}

#[derive(Copy, Clone)]
enum Material 
{
    Diffuse{ tint: Vector3 },
    Metal{ tint: Vector3 , fuzz: f32 },
    Emissive { tint: Vector3 }
}

impl Material  
{
    fn default() -> Material 
    {
        Material::Emissive{ tint: Vector3::new(1.0, 0.0, 1.0)}
    }

    fn DefaultSky() -> Material
    {
        return Material::Emissive{ tint: Vector3::new(0.9, 0.9, 0.9)}
    }
}

struct Sphere
{
    center: Vector3,
    radius: f32,

    material: Material
}

impl Hittable for Sphere
{
    fn hit(&self, min_max: [f32; 2], r: &Ray) -> Option<HitRecord>
    {
        // Vector expression of sphere:
        // ((point - s_center) dot (point - s_center)) = r^2 
        // Vector expression of a point on a ray
        // (origin + t * direction) = point
        // 
        //  Combine:
        //
        // (((origin + t * direction) - s_center) dot ((origin + t * direction)  - s_center)) - r^2 = 0 
        // 
        // all of these are ultimately scalars! so we can use quadratic formula to find solutions.
        //
        // polynomial representing ray hitting sphere in form of 
        // ax^2 + bx + c = 0
        // (dir dot dir) * t^2 + 2 * (dir dot distance) * t + ((dist dot dist) minus r squared) = 0
        // roots are:
        // (-b +- sqrt(b^2 - 4ac)) / 2a
        // discriminant is b^2 - 4ac

        let origin_to_center = r.position +- self.center;

        let a = r.direction.dot(r.direction);
        let b = 2.0 * r.direction.dot(origin_to_center);
        let c = origin_to_center.dot(origin_to_center) - self.radius * self.radius;

        let discriminant = b * b - 4.0 * a * c;

        if discriminant < 0.0 
        {
            None
        } 
        else 
        {
            let t_minus = (-b - discriminant.sqrt()) / (2.0 * a);
            let t_plus = (-b + discriminant.sqrt()) / (2.0 * a);

            let t_best = [t_minus, t_plus].into_iter().
                filter(|t| t > &min_max[0] && t < &min_max[1]).
                map(|f| FloatOrd(f)).
                min();

            if t_best.is_none() 
            {
                return None;
            }

            let hit_point = r.point_at(t_best.unwrap().0);
            let mut normal_unit = (hit_point + -self.center).unit();
            let mut front_face = true;

            if r.direction.dot(normal_unit) > 0.0 {
                // ray is inside the sphere
                normal_unit = -normal_unit;
                front_face = false;
            } 

            Some(HitRecord{distance: t_best.unwrap().0, hit_point, normal_unit, front_face, material: self.material})
        }
    }
}

struct HittableCollection 
{
    inner: Vec<Box<dyn Hittable>>,
    bounding_container: Box<dyn Hittable>
}

impl Hittable for HittableCollection
{
    fn hit(&self,  min_max: [f32; 2], r: &Ray) -> Option<HitRecord> 
    {
        if self.bounding_container.hit(min_max, r).is_some() 
        {
            let mut sub_bound = min_max;
            let mut best_hit: Option<HitRecord> = None;
            for child in self.inner.iter() 
            {
                let hit = child.hit(sub_bound, r);

                if hit.is_some() 
                {
                    sub_bound = [min_max[0], hit.unwrap().distance];
                    best_hit = hit.clone();
                }
            }
            return best_hit;

        } 
        else 
        {
            None
        }
    }
}

struct MyTriangle 
{
    pts : [Vector3; 3],
    normal: Vector3,

    material: Material,
}

impl Hittable for MyTriangle
{
    // equation for plane:
    // where p0 is a point on the plane and normal = normal;
    // all p such that
    // (p - p0) dot normal == 0

    // equation to point on ray:
    // r0 + t * dir = point

    // combine
    // (r0 + t*dir - p0) dot normal = 0
    // dot product is distributive
    //   (t * dir) dot normal =  (p0 - r0) dot normal
    // scalar and dot are distributive
    // t = (p0 - r0) dot normal / direction dot normal
    // if direction dot normal is 0, no intersection
    // if t is negative, solution is behind, exclude.
    
    //after you find plane collision point, check if it is in triangle

    fn hit(&self,  min_max: [f32; 2], r: &Ray) -> Option<HitRecord> 
    {
        let denominator = self.normal.dot(r.direction);

        if denominator < 0.0001 
        {
            return None;
        }

        let numerator = (self.pts[0] +- r.position).dot(self.normal);
        let t = numerator / denominator;
        if t < min_max[0] || t > min_max[1] 
        {
            return None;
        } 

        // inside outside test
        let p = r.point_at(t);

        let edge0 = self.pts[1] +- self.pts[0]; 
        let edge1 = self.pts[2] +- self.pts[1]; 
        let edge2 = self.pts[0] +- self.pts[2]; 
        let C0 = p +- self.pts[0]; 
        let C1 = p +- self.pts[1]; 
        let C2 = p +- self.pts[2]; 

        // if negative, that means the point is on the right side of the "side vector"
        // winding direction is important here.

        if self.normal.dot(edge0.cross(C0)) > 0.0 && 
            self.normal.dot(edge1.cross(C1)) > 0.0 && 
            self.normal.dot(edge2.cross(C2)) > 0.0
        {
            // TODO: front face?
            Some(HitRecord{distance: t, hit_point:p, normal_unit: self.normal.unit(), front_face: true, material: self.material})
        } 
        else 
        {
            return None;
        }      
    }
}

 // TODO: Rotation
fn build_model(stl_path: String, center: Vector3, bounding_sphere_dim: f32, mat: Material) -> HittableCollection
{
    let mut file = OpenOptions::new().read(true).open(stl_path.clone()).unwrap();
    let max_length = stl_io::create_stl_reader(&mut file).unwrap().
    flatten().
    flat_map(|t| t.vertices).
    map(|v| FloatOrd(Vector3::new(v[0], v[1], v[2]).length_squared())).
    max().
    unwrap();

    println!("max: {}", max_length.0);

    let ratio = bounding_sphere_dim / max_length.0.sqrt();

    let mut file  = OpenOptions::new().read(true).open(stl_path).unwrap();
    let triangles : Vec<Box<dyn Hittable>> = stl_io::create_stl_reader(&mut file).unwrap().
    flatten().
    map(|v| Box::new(convert(center, ratio, v, mat)) as Box<dyn Hittable>).
    collect();

    HittableCollection
    {
        inner: triangles,
        bounding_container : Box::new(Sphere{ center: center, radius: bounding_sphere_dim, material: mat })
    }
}

fn convert(center: Vector3, ratio: f32, their_triangle: stl_io::Triangle, mat: Material) -> MyTriangle 
{
    MyTriangle 
    {
        normal : Vector3::new(their_triangle.normal[0], their_triangle.normal[1],  their_triangle.normal[2]),
        pts: [  center+ ratio * Vector3::new(their_triangle.vertices[0][0], their_triangle.vertices[0][2], their_triangle.vertices[0][1]),
                center+ ratio * Vector3::new(their_triangle.vertices[1][0], their_triangle.vertices[1][2], their_triangle.vertices[1][1]),
                center+ ratio * Vector3::new(their_triangle.vertices[2][0], their_triangle.vertices[2][2], their_triangle.vertices[2][1])],
        material: mat
    }
}


fn build_world() -> HittableCollection 
{
    let elements: Vec<Box<dyn Hittable>> = vec!
    [
        //Box::new(Sphere{ center: Vector3::new(0.0, 0.0, 0.1), radius: 0.05 , material: Material::Diffuse{tint: Vector3::new(0.1, 0.1, 0.1)}}),

        Box::new(Sphere{ center: Vector3::new(-0.7, 0.0, -1.0), radius: 0.3 , material: Material::Metal{tint: Vector3::new(0.9, 0.9, 0.9), fuzz:0.1}}),
        //Box::new(Sphere{ center: Vector3::new(0.0, 0.0, -1.0), radius: 0.3 , material: Material::Diffuse{tint: Vector3::new(0.2, 0.4, 0.2)}}),
        Box::new(Sphere{ center: Vector3::new(0.7, 0.0, -1.0), radius: 0.3 , material: Material::Metal{tint: Vector3::new(0.9, 0.9, 0.9), fuzz:0.0}}),

        // Box::new(Sphere{ center: Vector3::new(0.0, 1.0, -0.7), radius: 0.05 , material: Material::Emissive{tint: Vector3::new(10.0, 10.0, 10.0)}}),
        // Box::new(Sphere{ center: Vector3::new(0.7, 1.0, -0.7), radius: 0.05 , material: Material::Emissive{tint: Vector3::new(10.0, 10.0, 10.0)}}),
        // Box::new(Sphere{ center: Vector3::new(0.7, 1.0, -0.0), radius: 0.05 , material: Material::Emissive{tint: Vector3::new(10.0, 10.0, 10.0)}}),
        // Box::new(Sphere{ center: Vector3::new(0.0, 1.0,  0.0), radius: 0.05 , material: Material::Emissive{tint: Vector3::new(10.0, 10.0, 10.0)}}),

        Box::new(Sphere{ center: Vector3::new(3.00, -0.3, -2.0), radius: 0.01 , material: Material::Emissive{tint: Vector3::new(10.0, 2.0, 2.0)}}),
        Box::new(Sphere{ center: Vector3::new(3.12, -0.3, -2.0), radius: 0.01 , material: Material::Emissive{tint: Vector3::new(10.0, 2.0, 2.0)}}),


        Box::new(Sphere{ center: Vector3::new(0.0, -101.5, -1.0), radius: 101.0, material: Material::Diffuse{tint: Vector3::new(0.6, 0.4, 0.4)}}),

        Box::new(build_model("X:\\Image_Testing\\human.stl".to_string(), Vector3::new(0.0, 0.0, -0.3), 0.7, Material::Diffuse{tint: Vector3::new(0.0, 0.0, 0.0)} ))
    ];

    let bound = Box::new(TrivialHitter{});

    return HittableCollection 
    {
        bounding_container: bound,
        inner: elements,
    }
}

fn get_subpixels_random(sample_count: i32, c: [i32; 2], sa: &SizeAspect, rand: &mut RandSource) -> Vec<[f32; 2]> 
{
    let u = c[0] as f32 / (sa.w_h[0]-1) as f32;
    let v = c[1] as f32 / (sa.w_h[1]-1) as f32;

    let increment_u = 1.0 / (sa.w_h[0]-1) as f32;
    let increment_v = 1.0 / (sa.w_h[1]-1) as f32;

    (1..=sample_count).into_iter().map(|_i| [u+rand.inner.gen::<f32>()*increment_u, v+rand.inner.gen::<f32>()*increment_v]).collect()
}

fn color_px(sample_count: i32, lerps: [f32; 2], c : [i32; 2], sa: &SizeAspect, world: &HittableCollection, rand: &mut RandSource) -> Vector3 
{
    let mut sum_vector = Vector3::BLACK();

    let subpixels = get_subpixels_random(sample_count, c, sa, rand);

    for subpixel in subpixels.iter()
    {
        sum_vector = sum_vector + color_px_subpixel(lerps, *subpixel, sa, world, rand);
    }

    let multisampled_vector = sum_vector * (1.0 / subpixels.len() as f32);

    // Color correction:
    // Right now only gamma corrected

    let color_corrected = 
    Vector3::new
    (
        multisampled_vector.r().sqrt(),
        multisampled_vector.g().sqrt(),
        multisampled_vector.b().sqrt(),
    );

    let clamped = 
    Vector3::new
    (
        clamp(color_corrected.r()),
        clamp(color_corrected.g()),
        clamp(color_corrected.b()),
    );

    return clamped;

}

fn color_px_subpixel(lerps: [f32; 2], c : [f32; 2], sa: &SizeAspect, world: &HittableCollection, rand: &mut RandSource) -> Vector3 
{
    let cam = CameraInfo::new(lerps, *sa);
    let parent_ray = Ray { position: cam.origin, direction: cam.lower_left_corner + cam.horizontal_vector*c[0] + cam.vertical_vector*c[1] + -cam.origin };

    return recursive_ray_bounce(lerps, [0.0001, 1000.0], &parent_ray, world, 4 * ((lerps[1] + 1.0) as i32), rand);
}

fn recursive_ray_bounce(lerps: [f32; 2], min_max: [f32;2], r : &Ray, world: &HittableCollection, bounces_remaining: i32, rand: &mut RandSource) -> Vector3 
{
    if bounces_remaining <= 0 
    {
       return Vector3::new(0.0, 0.0, 0.0);
    }

    match world.hit(min_max, &r) 
    {
        Some(h) => 
        {
            match h.material
            {   
                Material::Diffuse { tint } => 
                {
                    let random_lambertian =  h.normal_unit + rand.on_unit_sphere(); 
                    let child_ray = Ray { position: h.hit_point, direction : random_lambertian.unit() };
                    return tint * recursive_ray_bounce(lerps, min_max, &child_ray, world, bounces_remaining-1, rand);
                }
                Material::Metal { tint, fuzz } => 
                {
                    let metal_reflection_direction = r.direction.reflect_by(&h.normal_unit) + rand.on_unit_sphere() * fuzz;
                    let child_ray = Ray { position: h.hit_point, direction : metal_reflection_direction.unit() };
                    return tint * recursive_ray_bounce(lerps, min_max, &child_ray, world, bounces_remaining-1, rand);
                }
                Material::Emissive { tint } =>
                {
                    return tint;
                }
            }
        }
        None => 
        {
            match Material::DefaultSky() 
            {
                Material::Emissive { tint } => { return Vector3::new(0.2, 0.2, 0.2) }
                _ => { return Vector3::BLACK(); }
            }
        }
    }
}