use std::f32::MAX;

use dot_vox::{Model, Voxel};
use euc::{buffer::Buffer2d, rasterizer, Interpolate, Pipeline, Target};
use glam::Vec4Swizzles;
use maths_rs::prelude::*;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::Serialize;

fn glam_mat4_to_maths_mat4(mat: glam::Mat4) -> Mat4<f32> {
    let glam: [f32; 16] = mat.transpose().to_cols_array();
    Mat4::new(
        glam[0], glam[1], glam[2], glam[3], glam[4], glam[5], glam[6], glam[7], glam[8], glam[9],
        glam[10], glam[11], glam[12], glam[13], glam[14], glam[15],
    )
}

struct Cube<'a> {
    proj: Mat4<f32>,
    view: Mat4<f32>,
    model: Mat4<f32>,
    inverse_model: Mat4<f32>,
    camera_world_position: Vec4<f32>,
    voxel_dimensions: Vec3<u32>,
    voxel_texture: &'a [VoxelTexel],
    palette_texture: Vec<Vec4<f32>>,
    positions: &'a [Vec4<f32>],
}

fn sd_box(p: Vec3<f32>, b: Vec3<f32>) -> f32 {
    let q = abs(p) - b;
    length(max(q, Vec3f::zero())) + min(max(q.x, max(q.y, q.z)), 0.0)
}

fn sd_sphere(p: Vec3<f32>, s: f32) -> f32 {
    length(p) - s
}

impl<'a> Cube<'a> {
    fn intersect_aabb(
        &self,
        ray_origin: Vec3<f32>,
        ray_direction: Vec3<f32>,
        bounding_box_min: Vec3<f32>,
        bounding_box_max: Vec3<f32>,
    ) -> Vec2<f32> {
        let t_min = (bounding_box_min - ray_origin) / ray_direction;
        let t_max = (bounding_box_max - ray_origin) / ray_direction;

        let t1 = min(t_min, t_max);
        let t2 = max(t_min, t_max);

        let t_near = max(max(t1.x, t1.y), t1.z);
        let t_far = min(min(t2.x, t2.y), t2.z);

        Vec2::new(t_near, t_far)
    }
}

#[derive(Clone, Copy, Debug)]
struct VsOutWrapper {
    frag_origin: Vec3<f32>,
    frag_direction: Vec3<f32>,
}

impl Interpolate for VsOutWrapper {
    fn lerp2(a: Self, b: Self, x: f32, y: f32) -> Self {
        // a * x + b * y
        Self {
            frag_origin: a.frag_origin * x + b.frag_origin * y,
            frag_direction: a.frag_direction * x + b.frag_direction * y,
        }
    }
    fn lerp3(a: Self, b: Self, c: Self, x: f32, y: f32, z: f32) -> Self {
        // a * x + b * y
        Self {
            frag_origin: a.frag_origin * x + b.frag_origin * y + c.frag_origin * z,
            frag_direction: a.frag_direction * x + b.frag_direction * y + c.frag_direction * z,
        }
    }
}

impl<'a> Pipeline for Cube<'a> {
    type Vertex = usize;
    type VsOut = VsOutWrapper;
    type Pixel = u32;

    #[inline(always)]
    fn vert(&self, v_index: &Self::Vertex) -> ([f32; 4], Self::VsOut) {
        let mvp = self.proj * self.view * self.model * self.positions[*v_index];
        let camera_local = (self.inverse_model * self.camera_world_position).xyz();
        let frag_origin = camera_local;
        let frag_direction = self.positions[*v_index].xyz() - camera_local;

        (
            [mvp.x, mvp.y, mvp.z, mvp.w],
            VsOutWrapper {
                frag_direction,
                frag_origin,
            },
        )
    }

    #[inline(always)]
    fn frag(&self, frag_in: &Self::VsOut) -> Self::Pixel {
        let direction = normalize(frag_in.frag_direction);
        let mut point = frag_in.frag_origin;
        let bounding_box_min = Vec3::new(-0.5, -0.5, -0.5);
        let bounding_box_max = Vec3::new(0.5, 0.5, 0.5);

        point = point
            + direction
                * max(
                    0.0,
                    self.intersect_aabb(point, direction, bounding_box_min, bounding_box_max)
                        .x,
                );

        let scale = 2.0;
        point *= vec3f(
            self.voxel_dimensions.x as f32 / scale,
            self.voxel_dimensions.y as f32 / scale,
            self.voxel_dimensions.z as f32 / scale,
        );

        let voxel = self.voxel_texture.par_iter().find_any(|item| {
            if item.palette_index == 0 {
                return false;
            }
            let sdf_size = 0.5;
            let mut point_clone = point
                + ((vec3f(item.x as f32, item.y as f32, item.z as f32))
                    - vec3f(
                        self.voxel_dimensions.x as f32 / scale,
                        self.voxel_dimensions.y as f32 / scale,
                        self.voxel_dimensions.z as f32 / scale,
                    ));
            for _ in 0..30 {
                let distance = sd_box(point_clone, Vec3::new(sdf_size, sdf_size, sdf_size));
                if distance < 0.001 {
                    return true;
                }

                point_clone += direction * distance;
            }
            false
        });

        if let Some(item) = voxel {
            let bytes = self.palette_texture[item.palette_index as usize] * 255.0;
            return (bytes[2] as u32)
                | (bytes[1] as u32) << 8
                | (bytes[0] as u32) << 16
                | 0xFF << 24;
        }

        let bytes = Vec4::new(0.0, 0.0, 255.0, 255.0);
        (bytes[2] as u32)
            | (bytes[1] as u32) << 8
            | (bytes[0] as u32) << 16
            | (bytes[3] as u32) << 24
    }
}

const W: usize = 640;
const H: usize = 480;

#[derive(Copy, Clone, Default, Serialize, Debug)]
struct VoxelTexel {
    palette_index: u8,
    distance: f32,
    x: u8,
    y: u8,
    z: u8,
}

fn signed_distance(model: &Model, point: Vec3<f32>) -> f32 {
    let mut min_distance = std::f32::MAX;

    for voxel in &model.voxels {
        let voxel_center = Vec3::new(voxel.x as f32, voxel.y as f32, voxel.z as f32);
        let distance = sd_box(point - voxel_center, Vec3::new(0.2, 0.2, 0.2));
        min_distance = min(min_distance, distance);
    }

    min_distance
}

fn main() {
    let vox = dot_vox::load(r#"C:\Users\dylan\dev\cpu-voxel\castle.vox"#).unwrap();
    let model = vox.models.get(0).unwrap();

    let voxel_dimensions = Vec3::new(model.size.x, model.size.y, model.size.z);
    let mut voxel_texture: Vec<VoxelTexel> =
        vec![
            VoxelTexel::default();
            model.size.x as usize * model.size.y as usize * model.size.z as usize
        ];
    let mut palette_texture: Vec<Vec4<f32>> = vec![Vec4::new(0.0, 0.0, 0.0, 0.0); 257];

    println!("total len {}", voxel_texture.len());

    for x in 0..model.size.x {
        for y in 0..model.size.y {
            for z in 0..model.size.z {
                let i = x + y * model.size.x + z * model.size.x * model.size.y;

                if let Some(voxel) = model
                    .voxels
                    .iter()
                    .find(|v| v.x == x as u8 && v.y == y as u8 && v.z == z as u8)
                {
                    voxel_texture[i as usize].palette_index = voxel.i + 1;
                }
                voxel_texture[i as usize].x = x as u8;
                voxel_texture[i as usize].y = y as u8;
                voxel_texture[i as usize].z = z as u8;
            }
        }
    }

    let scale = 2.0;
    let direction = vec3f(0.0, 0.0, 1.0);
    let voxel = voxel_texture.par_iter().find_any(|item| {
        if item.palette_index == 0 {
            return false;
        }
        let sdf_size = 0.5;
        let mut point_clone = ((vec3f(item.x as f32, item.y as f32, item.z as f32))
            - vec3f(
                voxel_dimensions.x as f32 / scale,
                voxel_dimensions.y as f32 / scale,
                voxel_dimensions.z as f32 / scale,
            ));
        for _ in 0..30 {
            let distance = sd_box(point_clone, Vec3::new(sdf_size, sdf_size, sdf_size));
            if distance < 0.001 {
                return true;
            }

            point_clone += distance * direction;
        }
        false
    });

    serde_json::to_writer_pretty(
        std::fs::File::create(r#"./castle.json"#).unwrap(),
        &voxel_texture,
    )
    .unwrap();

    for (i, color) in vox.palette.iter().enumerate() {
        palette_texture[i + 1] = Vec4::new(
            color.r as f32 / 255.0,
            color.g as f32 / 255.0,
            color.b as f32 / 255.0,
            color.a as f32 / 255.0,
        );
    }

    let mut color = Buffer2d::new([W, H], 0);
    let mut depth = Buffer2d::new([W, H], 1.0);

    let mut win = minifb::Window::new(
        "Cpu voxel insanity",
        W,
        H,
        minifb::WindowOptions {
            resize: true,
            scale_mode: minifb::ScaleMode::AspectRatioStretch,
            ..Default::default()
        },
    )
    .unwrap();

    let camera_world_position = glam::Vec4::new(0.0, 0.0, 10.0, 1.0);
    let camera_target = glam::Vec3::new(0.0, 0.0, 0.0);

    println!("voxel_dimeonsions: {:?}", voxel_dimensions);
    println!("scale {}", (model.size.x as f32) / 8.0);
    let proj = glam_mat4_to_maths_mat4(glam::Mat4::perspective_rh(
        60_f32.to_radians(),
        W as f32 / H as f32,
        1.0,
        1000.0,
    ));
    let view = glam_mat4_to_maths_mat4(glam::Mat4::look_at_rh(
        camera_world_position.xyz(),
        camera_target,
        glam::Vec3::Y,
    ));

    let mut time_running = std::time::Instant::now();
    let mut now;
    now = std::time::Instant::now();
    let model = glam_mat4_to_maths_mat4(glam::Mat4::from_scale_rotation_translation(
        glam::Vec3::new(
            (model.size.x as f32) / 8.0,
            (model.size.y as f32) / 8.0,
            (model.size.z as f32) / 8.0,
        ),
        glam::Quat::from_rotation_x(-1.5),
        glam::Vec3::new(0.0, 0.0, -2.0),
    ));

    color.clear(0);
    depth.clear(1.0);

    Cube {
        proj,
        view,
        model,
        inverse_model: model.inverse(),
        voxel_texture: &voxel_texture,
        palette_texture: palette_texture.clone(),
        camera_world_position: Vec4::new(
            camera_world_position.x,
            camera_world_position.y,
            camera_world_position.z,
            1.0,
        ),
        voxel_dimensions,
        positions: &[
            Vec4::new(-1.0, -1.0, -1.0, 1.0), // 0
            Vec4::new(-1.0, -1.0, 1.0, 1.0),  // 1
            Vec4::new(-1.0, 1.0, -1.0, 1.0),  // 2
            Vec4::new(-1.0, 1.0, 1.0, 1.0),   // 3
            Vec4::new(1.0, -1.0, -1.0, 1.0),  // 4
            Vec4::new(1.0, -1.0, 1.0, 1.0),   // 5
            Vec4::new(1.0, 1.0, -1.0, 1.0),   // 6
            Vec4::new(1.0, 1.0, 1.0, 1.0),    // 7
        ],
    }
    .draw::<rasterizer::Triangles<_, rasterizer::BackfaceCullingDisabled>, _>(
        &[
            // -x
            0, 3, 2, 0, 1, 3, // +x
            7, 4, 6, 5, 4, 7, // -y
            5, 0, 4, 1, 0, 5, // +y
            2, 7, 6, 2, 3, 7, // -z
            0, 6, 4, 0, 2, 6, // +z
            7, 1, 5, 3, 1, 7,
        ],
        &mut color,
        Some(&mut depth),
    );

    for i in 0.. {
        if win.is_open() {
            win.update_with_buffer(color.as_ref(), W, H).unwrap();
        } else {
            break;
        }

        if time_running.elapsed().as_secs() == 2 {
            println!("fps: {}", 1.0 / now.elapsed().as_secs_f32());
            time_running = std::time::Instant::now();
        }
    }
}
