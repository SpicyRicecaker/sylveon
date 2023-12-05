use futures::executor;
use std::borrow::BorrowMut;

use anyhow::Error;
use glam::Vec3;
use wgpu::*;
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, KeyEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    raw_window_handle::{HasDisplayHandle, HasRawDisplayHandle, HasWindowHandle},
    window::WindowBuilder,
};

use log::{debug, error, info, log_enabled, Level};

struct Sphere {
    center: Vec3,
    radius: f32,
}

struct Camera {
    eye: Vec3,
    direction: Vec3,
    normal: Vec3,
    right: Vec3,
}

fn main() -> Result<(), Error> {
    // init logger
    env_logger::init();

    // ControlFlow::Wait pauses the event loop if no events are available to process.
    // This is ideal for non-game applications that only update in response to user
    // input, and uses significantly less power/CPU time than ControlFlow::Poll.
    // event_loop.set_control_flow(ControlFlow::Wait);

    // setup

    let spheres: Vec<Sphere> = vec![Sphere {
        center: Vec3::new(0., 0., -1.),
        radius: 0.5,
    }];

    let camera = Camera {
        eye: Vec3::new(0., 0., 0.),
        direction: Vec3::new(0., 0., -1.),
        normal: Vec3::new(0., 1., 0.),
        right: Vec3::new(1., 0., 0.),
    };

    let window_width: u32 = 400;
    // aspect ratio
    let aspect_ratio: f32 = 16. / 9.;
    let window_height: u32 = ((1. / aspect_ratio) * window_width as f32) as u32;

    // distance from camera eye to viewport
    let focal_length = 1.;

    let event_loop = EventLoop::new().unwrap();

    // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
    // dispatched any events. This is ideal for games and similar applications.
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = WindowBuilder::new()
        .with_inner_size(PhysicalSize::new(window_width, window_height))
        .build(&event_loop)
        .unwrap();

    // feel like everything past this point should be handled by wgpu

    // how do I create a shader that runs for every single pixel on the screen?
    // ans: i probaby don't, it's simply a compute shader.
    // how do I receive the output of a compute shader?

    // first let us clear the output of our screen using webgpu

    // get gpu
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::VULKAN,
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
    features.remove(
        Features::TEXTURE_COMPRESSION_ETC2
            | Features::TEXTURE_COMPRESSION_ASTC
            | Features::TEXTURE_COMPRESSION_ASTC_HDR
            | Features::VERTEX_ATTRIBUTE_64BIT
            | Features::SHADER_EARLY_DEPTH_TEST,
    );
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
            .get_default_config(&adapter, window_width, window_height)
            .unwrap(),
    );

    let screen_shader = std::fs::read_to_string("shaders/screen_shader.wgsl").unwrap();
    let raytracer_kernel = std::fs::read_to_string("shaders/raytracer_kernel.wgsl").unwrap();
    // create raytracing shader
    let screen_shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("screen shader"),
        source: ShaderSource::Wgsl(screen_shader.into()),
    });
    let raytracer_kernel = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("screen shader"),
        source: ShaderSource::Wgsl(raytracer_kernel.into()),
    });

    let color_buffer: Texture = device.create_texture(&TextureDescriptor {
        label: None,
        size: Extent3d {
            width: window_width,
            height: window_height,
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
    let ray_tracing_bind_group_layout =
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("raytracing bind group"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba8Unorm,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            }],
        });
    // what resources the raytracing pipeline will be using (just color buffer)
    let ray_tracing_bind_group: BindGroup = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &ray_tracing_bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::TextureView(&color_buffer_view),
        }],
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

    // hi
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
            label: None,
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

    // let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
    //     label: None,
    //     bind_group_layouts: &[],
    //     push_constant_ranges: &[],
    // });

    // let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
    //     label: Some("MAIN PIPE"),
    //     layout: None,
    //     vertex: (),
    //     primitive: (),
    //     depth_stencil: (),
    //     multisample: (),
    //     fragment: (),
    //     multiview: (),
    // });

    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent { event, .. } => {
                match event {
                    WindowEvent::RedrawRequested => {
                        // Redraw the application.
                        //
                        // It's preferable for applications that do not render continuously to render in
                        // this event rather than in AboutToWait, since rendering in here allows
                        // the program to gracefully handle redraws requested by the OS.
                    }
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                logical_key: key,
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    } => {
                        match key {
                            Key::Named(NamedKey::Escape) => {
                                elwt.exit();
                            }
                            _ => (),
                        }
                        dbg!(key);
                    }
                    WindowEvent::CloseRequested => {
                        println!("The close button was pressed; stopping");
                        elwt.exit();
                    }
                    _ => (),
                }
            }
            Event::AboutToWait => {
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
                            }
                        );
                        ray_tracing_compute_pass.set_bind_group(0, &ray_tracing_bind_group, &[]);
                        ray_tracing_compute_pass.set_pipeline(&ray_tracing_pipeline);
                        // the globalinvocation id is a vec3 that corresponds to current width and height
                        // TODO: how do we dispatch more than 1 ray per pixel (and randomly too)
                        // second, how do we update hit records for a sphere in the compute shader?
                        ray_tracing_compute_pass.dispatch_workgroups(
                            window_width,
                            window_height,
                            1,
                        );
                        // i think drop auto calls compute pass end
                    }
                    {
                        let render_pass_descriptor = RenderPassDescriptor {
                            label: None,
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
                        let mut screen_shader_render_pass = command_encoder.begin_render_pass(&render_pass_descriptor);
                        screen_shader_render_pass.set_bind_group(0, &screen_shader_bind_group, &[]);
                        screen_shader_render_pass.set_pipeline(&screen_shader_pipeline);
                        screen_shader_render_pass.draw(0..6, 0..1);
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
