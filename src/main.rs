use futures::executor::{self, block_on};
use hal::auxil::db;
use std::{
    borrow::BorrowMut,
    collections::{HashMap, HashSet, VecDeque},
    panic::AssertUnwindSafe,
};

use anyhow::Error;
use glam::{DVec2, UVec2, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    *,
};
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, KeyEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, ModifiersState, NamedKey, SmolStr},
    platform::modifier_supplement::KeyEventExtModifierSupplement,
    raw_window_handle::{HasDisplayHandle, HasRawDisplayHandle, HasWindowHandle},
    window::{CursorGrabMode, Window, WindowBuilder},
};

use log::{debug, error, info, log_enabled, Level};
use sylveon::{rasterizer::to_triangle_coords, *};

#[repr(C)]
#[derive(bytemuck::NoUninit, Debug, Clone, Copy, Default)]

struct Camera {
    eye: Vec3,
    focal_length: f32,
    direction: Vec3,
    aspect_ratio: f32,
    normal: Vec3,
    fov_y: f32,
    right: Vec3,
    _1: f32,
}

#[derive(Debug, Default)]
struct Keys {
    held_keys: HashSet<String>,
}

#[derive(Debug, Default)]

struct Scene {
    objects: Vec<Sphere>,
}

// for every 10 pixels, move camera 1 degree.
const DELTA_TO_DEGREES_RATIO: f64 = 0.1f64;
#[derive(Debug, Default)]
struct Mouse {
    rot: Vec2,
    delta: DVec2,
    active: bool,
}

#[derive(Debug, Default)]
struct Game {
    camera: Camera,
    window_size: UVec2,
    camera_mode: CameraMode,
    velocity: Vec3,
    keys: Keys,
    scene: Scene,
    mouse: Mouse,
}

impl Game {
    fn new(camera: Camera) -> Self {
        let width: u32 = 1000;
        // aspect ratio
        let height: u32 = ((1. / camera.aspect_ratio) * width as f32) as u32;

        Self {
            camera,
            camera_mode: CameraMode::Minecraft,
            keys: Keys::default(),
            scene: Scene {
                objects: vec![Sphere {
                    center: Vec3::new(0., 0., -1.),
                    radius: 0.5,
                    material: Material {
                        albedo: Vec3::new(1., 1., 1.),
                        padding_1: 0.,
                        ambient: Vec3::new(0., 0., 0.),
                        padding_2: 0.,
                    },
                }],
            },
            window_size: UVec2::new(width, height),
            mouse: Mouse::default(),
            ..Default::default()
        }
    }
}

impl Game {
    fn handle_key_event(&mut self, e: &KeyEvent, window: &Window) {
        match e.key_without_modifiers().as_ref() {
            Key::Named(NamedKey::Shift) => match e.state {
                ElementState::Released => {
                    self.keys.held_keys.remove("shift");
                }
                ElementState::Pressed => {
                    self.keys.held_keys.insert("shift".into());
                }
            },
            Key::Named(NamedKey::Space) => match e.state {
                ElementState::Released => {
                    self.keys.held_keys.remove("space");
                }
                ElementState::Pressed => {
                    self.keys.held_keys.insert("space".into());
                }
            },
            Key::Character(c) => match e.state {
                ElementState::Released => {
                    self.keys.held_keys.remove(c);
                }
                ElementState::Pressed => {
                    match c {
                        "f" => {
                            dbg!("switching to active mode");
                            if self.mouse.active {
                                window.set_cursor_grab(CursorGrabMode::None).unwrap();
                                window.set_cursor_visible(true);
                            } else {
                                window
                                    .set_cursor_grab(CursorGrabMode::Confined)
                                    .or_else(|_e| window.set_cursor_grab(CursorGrabMode::Locked))
                                    .unwrap();
                                window.set_cursor_visible(false);
                            }
                            self.mouse.active = !self.mouse.active;
                        }
                        "z" => {}
                        _ => {}
                    }
                    self.keys.held_keys.insert(c.into());
                }
            },
            _ => (),
        }
    }
    fn tick(&mut self) {
        // update camera
        {
            {
                let t = -DELTA_TO_DEGREES_RATIO * self.mouse.delta;
                self.mouse.rot.x += t.x as f32;
                self.mouse.rot.y += t.y as f32;
            }
            let model = glam::Mat3::from_rotation_y(self.mouse.rot.x.to_radians())
                * glam::Mat3::from_rotation_x(self.mouse.rot.y.to_radians());
            // let model = glam::Mat3::from_rotation_x(self.mouse.rot.y.to_radians()) * glam::Mat3::from_rotation_y(self.mouse.rot.x.to_radians());
            self.camera.direction = model.mul_vec3(Vec3::new(0., 0., -1.));
            self.camera.normal = model.mul_vec3(Vec3::new(0., 1., 0.));
            self.camera.right = model.mul_vec3(Vec3::new(1., 0., 0.));

            self.mouse.delta.x = 0.;
            self.mouse.delta.y = 0.;
        }

        // movement
        {
            if self.camera_mode == CameraMode::Minecraft {
                let mut v = Vec3::new(0., 0., 0.);

                let v_max: f32 = 0.01;
                if self.keys.held_keys.contains("w") {
                    v += v_max * self.camera.direction;
                }
                if self.keys.held_keys.contains("a") {
                    v -= v_max * self.camera.right;
                }
                if self.keys.held_keys.contains("s") {
                    v -= v_max * self.camera.direction;
                }
                if self.keys.held_keys.contains("d") {
                    v += v_max * self.camera.right;
                }
                if self.keys.held_keys.contains("space") {
                    v += v_max * Vec3::new(0., 1., 0.);
                }
                if self.keys.held_keys.contains("shift") {
                    v += v_max * Vec3::new(0., -1., 0.);
                }

                self.velocity = v;
            }

            self.camera.eye += self.velocity;
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum CameraMode {
    Minecraft,
    Blender,
}

impl Default for CameraMode {
    fn default() -> Self {
        Self::Minecraft
    }
}

fn main() -> Result<(), Error> {
    // init logger
    env_logger::init();

    // ControlFlow::Wait pauses the event loop if no events are available to process.
    // This is ideal for non-game applications that only update in response to user
    // input, and uses significantly less power/CPU time than ControlFlow::Poll.
    // event_loop.set_control_flow(ControlFlow::Wait);

    // setup
    let camera = Camera {
        eye: Vec3::new(0., 0., 0.),
        // direction: Vec3::new(0., 0., -1.),
        // normal: Vec3::new(0., 1., 0.),
        // right: Vec3::new(1., 0., 0.),
        focal_length: 1.,
        aspect_ratio: 16. / 9.,
        fov_y: 60.0,
        ..Default::default()
    };

    let mut game = Game::new(camera);

    let event_loop = EventLoop::new().unwrap();

    // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
    // dispatched any events. This is ideal for games and similar applications.
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = WindowBuilder::new()
        .with_inner_size(PhysicalSize::new(game.window_size.x, game.window_size.y))
        .build(&event_loop)
        .unwrap();

    // feel like everything past this point should be handled by wgpu

    // how do I create a shader that runs for every single pixel on the screen?
    // ans: i probaby don't, it's simply a compute shader.
    // how do I receive the output of a compute shader?

    // first let us clear the output of our screen using webgpu

    let backend: Backends = match std::env::consts::OS {
        "linux" => Backends::VULKAN,
        "macos" => Backends::METAL,
        "windows" => Backends::DX12,
        _ => {
            panic!("unsupported OS. Supported operating systems include: linux, macos, windows")
        }
    };

    // get gpu
    let instance = Instance::new(InstanceDescriptor {
        backends: backend,
        ..Default::default()
    });
    // get surface
    let surface = unsafe { instance.create_surface(&window) }.unwrap();
    // get adapter
    let adapter = executor::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .unwrap();
    // featureset supported on NVIDIA + Linux + Vulkan
    let mut features = Features::all();
    match backend {
        Backends::VULKAN => {
            features.remove(
                Features::TEXTURE_COMPRESSION_ETC2
                    | Features::TEXTURE_COMPRESSION_ASTC
                    | Features::TEXTURE_COMPRESSION_ASTC_HDR
                    | Features::VERTEX_ATTRIBUTE_64BIT
                    | Features::SHADER_EARLY_DEPTH_TEST,
            );
        }
        Backends::METAL => {
            features.remove(
                Features::PIPELINE_STATISTICS_QUERY
                    | Features::TIMESTAMP_QUERY_INSIDE_PASSES
                    | Features::BUFFER_BINDING_ARRAY
                    | Features::PARTIALLY_BOUND_BINDING_ARRAY
                    | Features::MULTI_DRAW_INDIRECT_COUNT
                    | Features::POLYGON_MODE_POINT
                    | Features::CONSERVATIVE_RASTERIZATION
                    | Features::SPIRV_SHADER_PASSTHROUGH
                    | Features::MULTIVIEW
                    | Features::SHADER_F64
                    | Features::SHADER_I16
                    | Features::VERTEX_ATTRIBUTE_64BIT
                    | Features::SHADER_EARLY_DEPTH_TEST,
            );
        }
        Backends::DX12 => features.remove(
            Features::SHADER_F16
                | Features::TEXTURE_COMPRESSION_ETC2
                | Features::TEXTURE_COMPRESSION_ASTC
                | Features::TEXTURE_COMPRESSION_ASTC_HDR
                | Features::PIPELINE_STATISTICS_QUERY
                | Features::BUFFER_BINDING_ARRAY
                | Features::STORAGE_RESOURCE_BINDING_ARRAY
                | Features::PARTIALLY_BOUND_BINDING_ARRAY
                | Features::POLYGON_MODE_POINT
                | Features::SPIRV_SHADER_PASSTHROUGH
                | Features::MULTIVIEW
                | Features::VERTEX_ATTRIBUTE_64BIT
                | Features::SHADER_UNUSED_VERTEX_OUTPUT
                | Features::SHADER_F64
                | Features::SHADER_I16
                | Features::SHADER_EARLY_DEPTH_TEST,
        ),
        _ => {
            unreachable!()
        }
    }
    // get device & queue
    let (device, queue) = executor::block_on(adapter.request_device(
        &DeviceDescriptor {
            label: None,
            features,
            limits: Limits::default(),
        },
        None,
    ))
    .unwrap();
    surface.configure(
        &device,
        &surface
            .get_default_config(&adapter, game.window_size.x, game.window_size.y)
            .unwrap(),
    );

    let rasterizer_shader = std::fs::read_to_string("shaders/rasterizer_shader.wgsl").unwrap();
    let screen_shader = std::fs::read_to_string("shaders/screen_shader.wgsl").unwrap();
    let raytracer_kernel = std::fs::read_to_string("shaders/raytracer_kernel.wgsl").unwrap();

    let rasterizer_shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("rasterizer shader"),
        source: ShaderSource::Wgsl(rasterizer_shader.into()),
    });
    // create raytracing shader
    let screen_shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("screen shader"),
        source: ShaderSource::Wgsl(screen_shader.into()),
    });
    let raytracer_kernel = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("screen shader"),
        source: ShaderSource::Wgsl(raytracer_kernel.into()),
    });

    let cam_uniform: Buffer = device.create_buffer(&BufferDescriptor {
        label: Some("cam uniform"),
        size: std::mem::size_of::<Camera>() as wgpu::BufferAddress,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::MAP_WRITE,
        mapped_at_creation: false,
    });

    let window_uniform: Buffer = device.create_buffer(&BufferDescriptor {
        label: Some("window size uniform"),
        size: std::mem::size_of::<UVec2>() as wgpu::BufferAddress,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::MAP_WRITE,
        mapped_at_creation: false,
    });

    let bush_system = OLSystem::new_bush_system();
    let generations = 5;
    let s = bush_system.generate(generations);
    let triangles = OLSystem::turtle(s);
    // let spheres = vec![];

    let (accel_struct, bounding_boxes) = AccelStruct::new(&triangles);

    let cubes = bounding_boxes
        .clone()
        .into_iter()
        .flat_map(to_triangle_coords)
        .collect::<Vec<_>>();
    // dbg!(triangles.len());
    // // last value is just for padding lol
    // let triangles: Vec<Triangle> = vec![Triangle {
    //     points: [
    //         Vec4::new(0., 1., -1., 0.),
    //         Vec4::new(1., 1., -1., 0.),
    //         Vec4::new(0., 2., -1., 0.),
    //     ],
    //     material: Material {
    //         albedo: Vec3::new(0.2, 0.8, 0.1),
    //         padding_1: 0.,
    //         ambient: Vec3::new(0., 0., 0.),
    //         padding_2: 0.,
    //     },
    // }];

    // init buffers
    let cubes_buffer: Buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bounding box buffer"),
        contents: bytemuck::cast_slice(&cubes),
        usage: BufferUsages::VERTEX,
    });

    let cubes_buffer_layout = VertexBufferLayout {
        array_stride: std::mem::size_of::<Vec3>() as wgpu::BufferAddress,
        step_mode: VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            format: VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        }],
    };

    let accel_struct_buffer: Buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("accel_structure_buffer"),
        contents: bytemuck::cast_slice(&accel_struct.tree),
        usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
    });

    let bounding_box_buffer: Buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bounding_box_buffer"),
        contents: bytemuck::cast_slice(&bounding_boxes),
        usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
    });

    let triangle_buffer: Buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("triangle_buffer"),
        contents: bytemuck::cast_slice(&triangles),
        usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
    });

    let color_buffer: Texture = device.create_texture(&TextureDescriptor {
        label: None,
        size: Extent3d {
            width: game.window_size.x,
            height: game.window_size.y,
            depth_or_array_layers: 1,
        },
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::COPY_DST
            | TextureUsages::COPY_SRC
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::STORAGE_BINDING,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        view_formats: &[],
    });
    let color_buffer_view: TextureView = color_buffer.create_view(&TextureViewDescriptor {
        ..Default::default()
    });

    let sampler: Sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("color buffer sampler"),
        address_mode_u: AddressMode::Repeat,
        address_mode_v: AddressMode::Repeat,
        mag_filter: FilterMode::Nearest,
        min_filter: FilterMode::Nearest,
        mipmap_filter: FilterMode::Nearest,
        ..Default::default()
    });

    // init raytracing pipeline
    let ray_tracing_bind_group_layout =
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("raytracing bind group"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    // what resources the raytracing pipeline will be using (just color buffer)
    let ray_tracing_bind_group: BindGroup = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &ray_tracing_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&color_buffer_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer(cam_uniform.as_entire_buffer_binding()),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Buffer(window_uniform.as_entire_buffer_binding()),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::Buffer(triangle_buffer.as_entire_buffer_binding()),
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::Buffer(bounding_box_buffer.as_entire_buffer_binding()),
            },
            BindGroupEntry {
                binding: 5,
                resource: BindingResource::Buffer(accel_struct_buffer.as_entire_buffer_binding()),
            },
        ],
    });

    let ray_tracing_pipeline_layout: PipelineLayout =
        device.create_pipeline_layout(&PipelineLayoutDescriptor {
            bind_group_layouts: &[&ray_tracing_bind_group_layout],
            ..Default::default()
        });

    let ray_tracing_pipeline: ComputePipeline =
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("raytracing compute pipeline"),
            layout: Some(&ray_tracing_pipeline_layout),
            module: &raytracer_kernel,
            entry_point: "main",
        });

    // init rasterizer pipeline
    let rasterizer_shader_bind_group_layout =
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("rasterizer shader bind group"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // BindGroupLayoutEntry {
                //     binding: 0,
                //     visibility: ShaderStages::VERTEX,
                //     ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                //     count: None,
                // },
                // BindGroupLayoutEntry {
                //     binding: 1,
                //     visibility: ShaderStages::FRAGMENT,
                //     ty: BindingType::Texture {
                //         sample_type: TextureSampleType::Float { filterable: false },
                //         view_dimension: TextureViewDimension::D2,
                //         multisampled: false,
                //     },
                //     count: None,
                // },
            ],
        });
    let rasterizer_shader_bind_group: BindGroup = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &rasterizer_shader_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(cam_uniform.as_entire_buffer_binding()),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer(window_uniform.as_entire_buffer_binding()),
            },
        ],
    });
    let rasterizer_shader_pipeline_layout: PipelineLayout =
        device.create_pipeline_layout(&PipelineLayoutDescriptor {
            bind_group_layouts: &[&rasterizer_shader_bind_group_layout],
            ..Default::default()
        });
    let rasterizer_shader_pipeline: RenderPipeline =
        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("rasterizer shader pipeline"),
            layout: Some(&rasterizer_shader_pipeline_layout),
            vertex: VertexState {
                module: &rasterizer_shader,
                entry_point: "vert_main",
                buffers: &[cubes_buffer_layout],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &rasterizer_shader,
                entry_point: "frag_main",
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::Bgra8UnormSrgb,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

    // init screen_shader pipeline
    let screen_shader_bind_group_layout =
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("screen shader bind group"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
    // what resources the raytracing pipeline will be using (just color buffer)
    let screen_shader_bind_group: BindGroup = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &screen_shader_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Sampler(&sampler),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&color_buffer_view),
            },
        ],
    });
    let screen_shader_pipeline_layout: PipelineLayout =
        device.create_pipeline_layout(&PipelineLayoutDescriptor {
            bind_group_layouts: &[&screen_shader_bind_group_layout],
            ..Default::default()
        });
    let screen_shader_pipeline: RenderPipeline =
        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("screen shader render pipeline"),
            layout: Some(&screen_shader_pipeline_layout),
            vertex: VertexState {
                module: &screen_shader,
                entry_point: "vert_main",
                buffers: &[],
            },
            primitive: PrimitiveState {
                // could use triangle strip here, but in anticipation
                // of future .obj files, holding off on that
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &screen_shader,
                entry_point: "frag_main",
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::Bgra8UnormSrgb,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

    let mut modifiers = ModifiersState::default();
    event_loop.run(move |event, elwt| {
        match event {
            Event::DeviceEvent { event, .. } => {
                if let winit::event::DeviceEvent::MouseMotion { delta } = event {
                    if game.mouse.active {
                        game.mouse.delta.x += delta.0;
                        game.mouse.delta.y += delta.1;
                    }
                }
            }
            Event::WindowEvent { event, .. } => {
                match event {
                    WindowEvent::ModifiersChanged(new) => {
                        modifiers = new.state();
                    }
                    WindowEvent::RedrawRequested => {
                        // Redraw the application.
                        //
                        // It's preferable for applications that do not render continuously to render in
                        // this event rather than in AboutToWait, since rendering in here allows
                        // the program to gracefully handle redraws requested by the OS.
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        game.handle_key_event(&event, &window);
                        if let KeyEvent {
                            logical_key: key,
                            state: ElementState::Pressed,
                            ..
                        } = &event
                        {
                            match key {
                                Key::Named(NamedKey::Escape) => {
                                    elwt.exit();
                                }
                                _ => {}
                            }
                        }
                    }
                    WindowEvent::CloseRequested => {
                        println!("The close button was pressed; stopping");
                        elwt.exit();
                    }
                    _ => (),
                }
            }
            Event::AboutToWait => {
                game.tick();

                cam_uniform.slice(..).map_async(MapMode::Write, |_| {});
                device.poll(MaintainBase::Wait);
                queue.write_buffer(&cam_uniform, 0, bytemuck::bytes_of(&game.camera));
                cam_uniform.unmap();

                window_uniform.slice(..).map_async(MapMode::Write, |_| {});
                device.poll(MaintainBase::Wait);
                queue.write_buffer(&window_uniform, 0, bytemuck::bytes_of(&game.window_size));
                window_uniform.unmap();

                let surface_texture = surface.get_current_texture().unwrap();
                let mut command_encoder =
                    device.create_command_encoder(&CommandEncoderDescriptor { label: None });
                let surface_texture_view =
                    surface_texture.texture.create_view(&TextureViewDescriptor {
                        ..Default::default()
                    });
                {
                    {
                        // Application update code.
                        let mut ray_tracing_compute_pass =
                            command_encoder.begin_compute_pass(&ComputePassDescriptor {
                                ..Default::default()
                            });
                        // update cam/window uniforms

                        ray_tracing_compute_pass.set_bind_group(0, &ray_tracing_bind_group, &[]);
                        ray_tracing_compute_pass.set_pipeline(&ray_tracing_pipeline);
                        // the globalinvocation id is a vec3 that corresponds to current width and height
                        // TODO: how do we dispatch more than 1 ray per pixel (and randomly too)
                        // second, how do we update hit records for a sphere in the compute shader?
                        ray_tracing_compute_pass.dispatch_workgroups(
                            game.window_size.x / 8,
                            game.window_size.y / 8,
                            1,
                        );
                        // i think drop auto calls compute pass end
                    }
                    {
                        let render_pass_descriptor = RenderPassDescriptor {
                            label: Some("render pass"),
                            color_attachments: &[Some(RenderPassColorAttachment {
                                view: &surface_texture_view,
                                resolve_target: None,
                                ops: Operations {
                                    load: LoadOp::Clear(Color {
                                        r: 0.1,
                                        g: 0.1,
                                        b: 0.1,
                                        a: 1.,
                                    }),
                                    store: StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        };
                        let mut screen_shader_render_pass =
                            command_encoder.begin_render_pass(&render_pass_descriptor);
                        screen_shader_render_pass.set_bind_group(0, &screen_shader_bind_group, &[]);
                        screen_shader_render_pass.set_pipeline(&screen_shader_pipeline);
                        screen_shader_render_pass.draw(0..6, 0..1);
                    }
                    {
                        let render_pass_descriptor = RenderPassDescriptor {
                            label: Some("rasterization pass"),
                            color_attachments: &[Some(RenderPassColorAttachment {
                                view: &surface_texture_view,
                                resolve_target: None,
                                ops: Operations {
                                    load: LoadOp::Load,
                                    store: StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        };

                        let mut rasterizer_shader_render_pass =
                            command_encoder.begin_render_pass(&render_pass_descriptor);
                        rasterizer_shader_render_pass.set_bind_group(
                            0,
                            &rasterizer_shader_bind_group,
                            &[],
                        );
                        rasterizer_shader_render_pass.set_pipeline(&rasterizer_shader_pipeline);

                        rasterizer_shader_render_pass.set_vertex_buffer(0, cubes_buffer.slice(..));
                        rasterizer_shader_render_pass.draw(0..(cubes.len() as u32), 0..1);
                    }
                }

                queue.submit(std::iter::once(command_encoder.finish()));
                surface_texture.present();

                // Queue a RedrawRequested event.
                //
                // You only need to call this if you've determined that you need to redraw, in
                // applications which do not always need to. Applications that redraw continuously
                // can just render here instead.
                window.request_redraw();
            }
            _ => (),
        }
    })?;

    Ok(())
}
