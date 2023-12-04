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

                    // Application update code.
                    let render_pass = command_encoder.begin_render_pass(&render_pass_descriptor);
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
