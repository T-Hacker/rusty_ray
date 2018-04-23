#![feature(test)]
extern crate vek;

use std::boxed::Box;
use std::f32::INFINITY;
use std::fs::File;
use std::io::{Result, Write};

use vek::vec::repr_simd::Vec3;
use vek::Clamp;

type F = f32;
type Vec = Vec3<F>;

const MAX_RAY_DEPTH: usize = 5;
const WIDTH: usize = 640;
const HEIGHT: usize = 480;
const IMAGE_BUFFER_SIZE: usize = WIDTH * HEIGHT * 3;

trait VisualObject {
    fn transparency(&self) -> F;
    fn reflection(&self) -> F;
    fn surface_color(&self) -> Vec;
    fn emission_color(&self) -> Vec;

    fn intersect(&self, ray_origin: Vec, ray_direction: Vec) -> Option<(F, F)>;
    fn calculate_normal(&self, intersection: Vec) -> Vec;
    fn calculate_light_direction(&self, intersection: Vec) -> Vec;
}

struct Sphere {
    center: Vec,
    surface_color: Vec,
    emission_color: Vec,

    radius_2: F,
    transparency: F,
    reflection: F,
}

impl Sphere {
    fn new(
        center: Vec,
        radius: F,
        surface_color: Vec,
        reflection: F,
        transparency: F,
        emission_color: Vec,
    ) -> Sphere {
        let radius2 = radius * radius;

        Sphere {
            center: center,
            surface_color: surface_color,
            emission_color: emission_color,
            radius_2: radius2,
            transparency: transparency,
            reflection: reflection,
        }
    }
}

impl VisualObject for Sphere {
    fn transparency(&self) -> F {
        self.transparency
    }

    fn reflection(&self) -> F {
        self.reflection
    }

    fn surface_color(&self) -> Vec {
        self.surface_color
    }

    fn emission_color(&self) -> Vec {
        self.emission_color
    }

    fn intersect(&self, ray_origin: Vec, ray_direction: Vec) -> Option<(F, F)> {
        let l = self.center - ray_origin;
        let tca = l.dot(ray_direction);
        if tca < 0.0f32 {
            return None;
        }

        let d2 = l.dot(l) - tca * tca;
        if d2 > self.radius_2 {
            return None;
        }

        let thc = (self.radius_2 - d2).sqrt();

        Some((tca - thc, tca + thc))
    }

    fn calculate_normal(&self, intersection: Vec) -> Vec {
        let mut normal = intersection - self.center;
        Vec::normalize(&mut normal);

        normal
    }

    fn calculate_light_direction(&self, intersection: Vec) -> Vec {
        let mut light_direction = self.center - intersection;
        Vec::normalize(&mut light_direction);

        light_direction
    }
}

fn mix(a: F, b: F, mix: F) -> F {
    b * mix + a * (1.0f32 - mix)
}

fn trace(ray_origin: Vec, ray_direction: Vec, objects: &[Box<VisualObject>], depth: usize) -> Vec {
    let mut tnear = INFINITY;

    let mut object: Option<&Box<VisualObject>> = None;

    // find intersection of this ray with the sphere in the scene
    for obj in objects {
        if let Some((mut t0, t1)) = obj.intersect(ray_origin, ray_direction) {
            if t0 < 0.0f32 {
                t0 = t1;
            }

            if t0 < tnear {
                tnear = t0;
                object = Some(obj);
            }
        }
    }

    // if there's no intersection return black or background color
    if let Some(obj) = object {
        let mut surface_color = Vec::zero(); // color of the ray/surfaceof the object intersected by the ray
        let p_hit = ray_origin + ray_direction * tnear; // point of intersection
        let mut n_hit = obj.calculate_normal(p_hit);

        // If the normal and the view direction are not opposite to each other
        // reverse the normal direction. That also means we are inside the sphere so set
        // the inside bool to true. Finally reverse the sign of IdotN which we want
        // positive.

        let bias = 1e-4f32;
        let mut inside = false;

        if ray_direction.dot(n_hit) > 0.0f32 {
            n_hit = -n_hit;
            inside = true;
        }

        if (obj.transparency() > 0.0f32 || obj.reflection() > 0.0f32) && depth < MAX_RAY_DEPTH {
            let facing_ratio = -ray_direction.dot(n_hit);

            // change the mix value to tweak the effect
            let fresnel_effect = mix((1.0f32 - facing_ratio).powf(3.0f32), 1.0f32, 0.1f32);

            // compute reflection direction (not need to normalize because all vectors are already normalized)
            let mut reflection_dir = ray_direction - n_hit * 2.0f32 * ray_direction.dot(n_hit);
            Vec::normalize(&mut reflection_dir);

            let reflection = trace(p_hit + n_hit * bias, reflection_dir, objects, depth + 1);
            let mut refraction = Vec::zero();

            // if the sphere is also transparent compute refraction ray (transmission)
            if obj.transparency() > 0.0f32 {
                let ior = 1.1f32;
                let eta = if inside { ior } else { 1.0f32 / ior };
                let cosi = -n_hit.dot(ray_direction);
                let k = 1.0f32 - eta * eta * (1.0f32 - cosi * cosi);
                let mut refraction_dir = ray_direction * eta + n_hit * (eta * cosi - k.sqrt());
                Vec::normalize(&mut refraction_dir);

                refraction = trace(p_hit - n_hit * bias, refraction_dir, objects, depth + 1);
            }

            surface_color = (reflection * fresnel_effect
                + refraction * (1.0f32 - fresnel_effect) * obj.transparency())
                * obj.surface_color();
        } else {
            // it's a diffuse object, no need to raytrace any further
            let mut i = 0;
            for o in objects {
                if o.emission_color().x > 0.0f32 {
                    // this is a light
                    let mut transmission = Vec::one();
                    let light_direction = o.calculate_light_direction(p_hit);

                    let mut j = 0;
                    for o2 in objects {
                        if i != j {
                            if let Some((_t0, _t1)) =
                                o2.intersect(p_hit + n_hit * bias, light_direction)
                            {
                                transmission = Vec::zero();
                                break;
                            }
                        }

                        j += 1;
                    }

                    surface_color += obj.surface_color() * transmission
                        * n_hit.dot(light_direction).max(0.0f32)
                        * o.emission_color();
                }

                i += 1;
            }
        }

        surface_color + obj.emission_color()
    } else {
        Vec::broadcast(2.0f32)
    }
}

fn render(objects: &[Box<VisualObject>]) -> std::vec::Vec<Vec> {
    let mut image = std::vec::Vec::with_capacity(WIDTH * HEIGHT);

    let width = WIDTH as F;
    let height = HEIGHT as F;
    let inv_width = 1.0f32 / width;
    let inv_height = 1.0f32 / height;
    let fov = 30.0f32;
    let aspect_ratio = width / height;
    let angle = (std::f32::consts::PI * 0.5f32 * fov / 180.0f32).tan();

    // Trace rays.
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let xx = (2.0f32 * (((x as F) + 0.5f32) * inv_width) - 1.0f32) * angle * aspect_ratio;
            let yy = (1.0f32 - 2.0f32 * (((y as F) + 0.5f32) * inv_height)) * angle;
            let mut ray_dir = Vec::new(xx, yy, -1.0f32);
            Vec::normalize(&mut ray_dir);

            image.push(trace(Vec::zero(), ray_dir, objects, 0));
        }
    }

    image
}

fn write_image(filename: &str, width: usize, height: usize, buffer: &[u8]) -> Result<()> {
    assert_eq!(
        width * height * 3,
        buffer.len(),
        "Buffer with invalid size."
    );

    let mut file = File::create(filename)?;
    file.write_all(format!("P6 {} {} 255 ", width, height).as_bytes())?;
    file.write_all(buffer)?;

    Ok(())
}

fn create_scene() -> std::vec::Vec<Box<VisualObject>> {
    let mut objects: std::vec::Vec<Box<VisualObject>> = std::vec::Vec::with_capacity(6);

    // Objects.
    objects.push(Box::new(Sphere::new(
        Vec::new(0.0f32, -10004.0f32, -20.0f32),
        10000.0f32,
        Vec::new(0.2f32, 0.2f32, 0.2f32),
        0.0f32,
        0.0f32,
        Vec::zero(),
    )));

    objects.push(Box::new(Sphere::new(
        Vec::new(0.0f32, 0.0f32, -20.0f32),
        4.0f32,
        Vec::new(1.0f32, 0.32f32, 0.36f32),
        1.0f32,
        0.5f32,
        Vec::zero(),
    )));

    objects.push(Box::new(Sphere::new(
        Vec::new(5.0f32, -1.0f32, -15.0f32),
        2.0f32,
        Vec::new(0.9f32, 0.76f32, 0.49f32),
        1.0f32,
        0.0f32,
        Vec::zero(),
    )));

    objects.push(Box::new(Sphere::new(
        Vec::new(5.0f32, 0.0f32, -25.0f32),
        3.0f32,
        Vec::new(0.65f32, 0.77f32, 0.97f32),
        1.0f32,
        0.0f32,
        Vec::zero(),
    )));

    objects.push(Box::new(Sphere::new(
        Vec::new(-5.5f32, 0.0f32, -15.0f32),
        3.0f32,
        Vec::new(0.9f32, 0.9f32, 0.9f32),
        1.0f32,
        0.0f32,
        Vec::zero(),
    )));

    // Lights.
    objects.push(Box::new(Sphere::new(
        Vec::new(0.0f32, 20.0f32, -30.0f32),
        3.0f32,
        Vec::new(0.0f32, 0.0f32, 0.0f32),
        0.0f32,
        0.0f32,
        Vec::broadcast(3.0f32),
    )));

    objects
}

fn main() -> std::io::Result<()> {
    let objects = create_scene();

    let image = render(&objects[..]);
    let mut buffer: [u8; IMAGE_BUFFER_SIZE] = [0; IMAGE_BUFFER_SIZE];
    let mut i = 0;
    for mut px in image {
        px = px.clamped(Vec::broadcast(0.0f32), Vec::broadcast(1.0f32));
        px *= Vec::broadcast(255.0f32);

        buffer[i] = px.x as u8;
        i += 1;

        buffer[i] = px.y as u8;
        i += 1;

        buffer[i] = px.z as u8;
        i += 1;
    }

    write_image("frame.ppm", WIDTH, HEIGHT, &buffer)?;

    Ok(())
}

// Benchmark code.
extern crate test;

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_render(b: &mut Bencher) {
        let objects = create_scene();

        b.iter(|| render(&objects[..]))
    }
}
