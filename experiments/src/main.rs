use std::{array, io, path::PathBuf, usize};

use egui::{
    emath::TSTransform, load::SizedTexture, output, CentralPanel, Color32, Pos2, Stroke,
    TextureHandle, TextureOptions, Ui, Vec2,
};
use egui_snarl::{
    ui::{PinInfo, SnarlViewer, SnarlWidget},
    InPin, OutPin, Snarl,
};
use jkl::{
    bc1,
    bits::WriteBits,
    encode::{FixedCode, VarCode},
    image::{ImageMut, ImageRef},
    lz77,
    math::{interleave16_2, Rgb32F, Rgb565, Rgb8U, Rgba8U, Vec3},
    max_rects::MaximalRectangles,
    reference_map::ReferenceMap,
    rle::{rle_with_cfg, RleCfg},
    vle::{self, Vle},
    zigzaq::ZigZag,
};

fn main() {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Jackal Experiments",
        native_options,
        Box::new(|cc| Ok(Box::new(Jackal::new(cc)))),
    )
    .unwrap();
}

struct Jackal {
    snarl: Snarl<JackalNode>,
    to_global: TSTransform,
}

impl Jackal {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut snarl = Snarl::new();
        if let Some(storage) = cc.storage {
            if let Some(data) = storage.get_string("snarl") {
                match serde_json::from_str(&data) {
                    Ok(deserialized) => {
                        snarl = deserialized;
                    }
                    Err(e) => {
                        eprintln!("Failed to deserialize snarl: {}", e);
                    }
                }
            }
        }
        Jackal {
            snarl,
            to_global: TSTransform::IDENTITY,
        }
    }
}

impl eframe::App for Jackal {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.snarl.nodes_mut().for_each(|node| node.prepare(ctx));

        if let Some(dropped_file) = ctx.input(|i| i.raw.dropped_files.last().cloned()) {
            if let Some(path) = &dropped_file.path {
                if let Ok(image) = image::open(path) {
                    let node = SourceImageNode {
                        file: path.to_path_buf(),
                        image: Ok(convert_image(image)),
                        body: ImageWidget::new(),
                    };

                    let pos = match ctx.input(|i| i.pointer.latest_pos()) {
                        None => Pos2::ZERO,
                        Some(pos) => self.to_global.inverse() * pos,
                    };
                    self.snarl.insert_node(pos, JackalNode::SourceImage(node));
                }
            }
        }

        CentralPanel::default().show(ctx, |ui| {
            let mut viewer = JackalViewer {
                to_global: self.to_global,
            };
            SnarlWidget::new().show(&mut self.snarl, &mut viewer, ui);
            self.to_global = viewer.to_global;
        });
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        match serde_json::to_string(&self.snarl) {
            Ok(serialized) => {
                storage.set_string("snarl", serialized);
            }
            Err(e) => {
                eprintln!("Failed to serialize snarl: {}", e);
            }
        }
    }
}

struct JackalViewer {
    to_global: TSTransform,
}

impl SnarlViewer<JackalNode> for JackalViewer {
    fn title(&mut self, node: &JackalNode) -> String {
        node.title()
    }

    fn inputs(&mut self, node: &JackalNode) -> usize {
        node.inputs()
    }

    #[allow(refining_impl_trait)]
    fn show_input(&mut self, pin: &InPin, ui: &mut Ui, snarl: &mut Snarl<JackalNode>) -> PinInfo {
        let node = &mut snarl[pin.id.node];
        node.input_ui(pin.id.input, ui);

        PinInfo::circle()
            .with_stroke(Stroke::new(1.0, Color32::WHITE))
            .with_fill(node.input_ty(pin.id.input).color())
    }

    fn outputs(&mut self, node: &JackalNode) -> usize {
        node.outputs()
    }

    #[allow(refining_impl_trait)]
    fn show_output(&mut self, pin: &OutPin, ui: &mut Ui, snarl: &mut Snarl<JackalNode>) -> PinInfo {
        let node = &mut snarl[pin.id.node];
        ui.add_space(ui.spacing().item_spacing.x);
        node.output_ui(pin.id.output, ui);

        PinInfo::circle()
            .with_stroke(Stroke::new(1.0, Color32::WHITE))
            .with_fill(node.output_ty(pin.id.output).color())
    }

    fn has_body(&mut self, node: &JackalNode) -> bool {
        node.has_body()
    }

    fn show_body(
        &mut self,
        node: egui_snarl::NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut Ui,
        snarl: &mut Snarl<JackalNode>,
    ) {
        let node = &mut snarl[node];
        node.body_ui(ui);
    }

    fn connect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<JackalNode>) {
        let from_node = &snarl[from.id.node];
        let ty = from_node.output_ty(from.id.output);

        let to_node = &mut snarl[to.id.node];
        let accepted = to_node.set_input_ty(to.id.input, ty);

        if accepted {
            let from_node = &snarl[from.id.node];
            let data = from_node.get_output(from.id.output);

            let to_node = &mut snarl[to.id.node];
            to_node.set_input(to.id.input, data);

            snarl.connect(from.id, to.id);
        }
    }

    fn disconnect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<JackalNode>) {
        let to_node = &mut snarl[to.id.node];
        to_node.set_input_ty(to.id.input, JackalType::Null);
        to_node.set_input(to.id.input, JackalValue::Null);
        snarl.disconnect(from.id, to.id);
    }

    fn current_transform(&mut self, to_global: &mut TSTransform, _snarl: &mut Snarl<JackalNode>) {
        self.to_global = *to_global;
    }

    fn has_node_menu(&mut self, _node: &JackalNode) -> bool {
        true
    }

    fn show_node_menu(
        &mut self,
        node: egui_snarl::NodeId,
        _inputs: &[InPin],
        outputs: &[OutPin],
        ui: &mut Ui,
        snarl: &mut Snarl<JackalNode>,
    ) {
        let r = ui.button("Delete");
        let r = r.on_hover_text("Delete this node");
        if r.clicked() {
            snarl.remove_node(node);

            for output in outputs {
                for remote in &output.remotes {
                    let remote_node = &mut snarl[remote.node];
                    remote_node.set_input_ty(remote.input, JackalType::Null);
                    remote_node.set_input(remote.input, JackalValue::Null);
                }
            }
        }
    }

    fn has_graph_menu(&mut self, _pos: Pos2, _snarl: &mut Snarl<JackalNode>) -> bool {
        true
    }

    fn show_graph_menu(&mut self, pos: Pos2, ui: &mut Ui, snarl: &mut Snarl<JackalNode>) {
        ui.vertical(|ui| {
            ui.menu_button("Add Filter Node", |ui| {
                let r = ui.button("Paeth");
                let r = r.on_hover_text("Add a Paeth image filter node");
                if r.clicked() {
                    snarl.insert_node(pos, JackalNode::Filter(FilterNode::new(Filter::Paeth)));
                }

                let r = ui.button("GammaG");
                let r = r.on_hover_text("Add a Gamma G image filter node");
                if r.clicked() {
                    snarl.insert_node(pos, JackalNode::Filter(FilterNode::new(Filter::GammaG)));
                }

                let r = ui.button("BC1");
                let r = r.on_hover_text("Add a BC1 image filter node");
                if r.clicked() {
                    snarl.insert_node(pos, JackalNode::Filter(FilterNode::new(Filter::BC1)));
                }
            });

            let r = ui.button("Add Strip Alpha Node");
            let r = r.on_hover_text("Add an Strip Alpha filter node");
            if r.clicked() {
                snarl.insert_node(pos, JackalNode::Filter(FilterNode::new(Filter::StripAlpha)));
            }

            let r = ui.button("Add Size Of Node");
            let r = r.on_hover_text("Add an Size Of node");
            if r.clicked() {
                snarl.insert_node(pos, JackalNode::SizeOf(SizeOfNode::new()));
            }

            let r = ui.button("Add LZP calculator Node");
            let r = r.on_hover_text("Add an LZP calculator node");
            if r.clicked() {
                snarl.insert_node(pos, JackalNode::LZPCalculator(LZPCalculatorNode::new()));
            }

            let r = ui.button("Add LZ77 calculator Node");
            let r = r.on_hover_text("Add an LZ77 calculator node");
            if r.clicked() {
                snarl.insert_node(pos, JackalNode::LZ77Calculator(LZ77CalculatorNode::new()));
            }

            let r = ui.button("Add LZ78 calculator Node");
            let r = r.on_hover_text("Add an LZ78 calculator node");
            if r.clicked() {
                snarl.insert_node(pos, JackalNode::LZ78Calculator(LZ78CalculatorNode::new()));
            }

            let r = ui.button("Add rANS calculator Node");
            let r = r.on_hover_text("Add an rANS calculator node");
            if r.clicked() {
                snarl.insert_node(pos, JackalNode::RansCalculator(RansCalculatorNode::new()));
            }

            let r = ui.button("Add LZ77+rANS calculator Node");
            let r = r.on_hover_text("Add an LZ77+rANS calculator node");
            if r.clicked() {
                snarl.insert_node(
                    pos,
                    JackalNode::LZ77RansCalculator(LZ77RansCalculatorNode::new()),
                );
            }

            let r = ui.button("Add LZ78+rANS calculator Node");
            let r = r.on_hover_text("Add an LZ78+rANS calculator node");
            if r.clicked() {
                snarl.insert_node(
                    pos,
                    JackalNode::LZ78RansCalculator(LZ78RansCalculatorNode::new()),
                );
            }

            let r = ui.button("Add RLE+rANS calculator Node");
            let r = r.on_hover_text("Add an RLE+rANS calculator node");
            if r.clicked() {
                snarl.insert_node(
                    pos,
                    JackalNode::RleRansCalculator(RleRansCalculatorNode::new()),
                );
            }

            let r = ui.button("Add Atlas Node");
            let r = r.on_hover_text("Add an Atlas node");
            if r.clicked() {
                snarl.insert_node(pos, JackalNode::Atlas(AtlasNode::new()));
            }

            let r = ui.button("Add Reference Map Node");
            let r = r.on_hover_text("Add an Reference Map node");
            if r.clicked() {
                snarl.insert_node(pos, JackalNode::ReferenceMap(ReferenceMapNode::new()));
            }

            let r = ui.button("Add Block Copy Node");
            let r = r.on_hover_text("Add a Block Copy node");
            if r.clicked() {
                snarl.insert_node(pos, JackalNode::BlockCopyNode(BlockCopyNode::new()));
            }
        });
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PixelType {
    Rgb8U,
    Rgba8U,
    BC1,
}

impl PixelType {
    fn name(&self) -> &'static str {
        match *self {
            PixelType::Rgb8U => "Rgb8U",
            PixelType::Rgba8U => "Rgba8U",
            PixelType::BC1 => "BC1",
        }
    }

    fn default(&self) -> PixelValue {
        match self {
            PixelType::Rgb8U => PixelValue::Rgb8U(Rgb8U::BLACK),
            PixelType::Rgba8U => PixelValue::Rgba8U(Rgba8U::BLACK),
            PixelType::BC1 => PixelValue::BC1(bc1::Block::BLACK),
        }
    }

    fn bit_size(&self) -> usize {
        match self {
            PixelType::Rgb8U => 24,
            PixelType::Rgba8U => 32,
            PixelType::BC1 => 64,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JackalType {
    Null,
    Uint,
    Pixel(PixelType),
    Image(PixelType),
}

impl JackalType {
    fn color(&self) -> Color32 {
        match *self {
            JackalType::Null => Color32::PLACEHOLDER,
            JackalType::Uint => Color32::RED,
            JackalType::Pixel(PixelType::Rgb8U) => Color32::BLUE,
            JackalType::Pixel(PixelType::Rgba8U) => Color32::GREEN,
            JackalType::Pixel(PixelType::BC1) => Color32::YELLOW,
            JackalType::Image(PixelType::Rgb8U) => Color32::LIGHT_BLUE,
            JackalType::Image(PixelType::Rgba8U) => Color32::LIGHT_GREEN,
            JackalType::Image(PixelType::BC1) => Color32::LIGHT_YELLOW,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum PixelValue {
    Rgb8U(Rgb8U),
    Rgba8U(Rgba8U),
    BC1(bc1::Block),
}

impl PixelValue {
    fn pixel_ty(&self) -> PixelType {
        match *self {
            PixelValue::Rgb8U(_) => PixelType::Rgb8U,
            PixelValue::Rgba8U(_) => PixelType::Rgba8U,
            PixelValue::BC1(_) => PixelType::BC1,
        }
    }

    fn ty(&self) -> JackalType {
        JackalType::Pixel(self.pixel_ty())
    }

    fn hash(&self) -> usize {
        match self {
            PixelValue::Rgb8U(pixel) => {
                (pixel.r() as usize) * 43 + (pixel.g() as usize) * 31 + (pixel.b() as usize) * 29
            }
            PixelValue::Rgba8U(pixel) => {
                (pixel.r() as usize) * 43
                    + (pixel.g() as usize) * 31
                    + (pixel.b() as usize) * 29
                    + (pixel.a() as usize) * 83
            }
            PixelValue::BC1(block) => {
                block.color0.bits() as usize * 43
                    + block.color1.bits() as usize * 31
                    + u32::from_le_bytes(block.texels) as usize
            }
        }
    }

    pub fn rgb(&self) -> Rgb8U {
        match *self {
            PixelValue::Rgb8U(p) => p,
            PixelValue::Rgba8U(p) => p.rgb(),
            PixelValue::BC1(b) => b.color0.into_8u(),
        }
    }

    pub fn rgba(&self) -> Rgba8U {
        match *self {
            PixelValue::Rgb8U(p) => p.into_opaque(),
            PixelValue::Rgba8U(p) => p,
            PixelValue::BC1(b) => b.color0.into_8u().into_opaque(),
        }
    }
}

#[derive(Clone)]
enum ImageValue {
    Rgb8U(Image<Rgb8U>),
    Rgba8U(Image<Rgba8U>),
    BC1(Image<bc1::Block>),
}

impl ImageValue {
    pub fn new(width: usize, height: usize, pixel_type: PixelType) -> Self {
        match pixel_type {
            PixelType::Rgb8U => ImageValue::Rgb8U(Image::solid(width, height, Rgb8U::BLACK)),
            PixelType::Rgba8U => ImageValue::Rgba8U(Image::solid(width, height, Rgba8U::BLACK)),
            PixelType::BC1 => ImageValue::BC1(Image::solid(width, height, bc1::Block::BLACK)),
        }
    }

    fn pixel_ty(&self) -> PixelType {
        match *self {
            ImageValue::Rgb8U(_) => PixelType::Rgb8U,
            ImageValue::Rgba8U(_) => PixelType::Rgba8U,
            ImageValue::BC1(_) => PixelType::BC1,
        }
    }

    fn ty(&self) -> JackalType {
        JackalType::Image(self.pixel_ty())
    }

    fn pixel_name(&self) -> &'static str {
        self.pixel_ty().name()
    }

    fn width(&self) -> usize {
        match self {
            ImageValue::Rgb8U(image) => image.width,
            ImageValue::Rgba8U(image) => image.width,
            ImageValue::BC1(image) => image.width,
        }
    }

    fn height(&self) -> usize {
        match self {
            ImageValue::Rgb8U(image) => image.height,
            ImageValue::Rgba8U(image) => image.height,
            ImageValue::BC1(image) => image.height,
        }
    }

    fn to_egui(&self) -> egui::ColorImage {
        match self {
            ImageValue::Rgb8U(image) => image.to_egui(),
            ImageValue::Rgba8U(image) => image.to_egui(),
            ImageValue::BC1(image) => image.to_egui(),
        }
    }

    fn get(&self, x: usize, y: usize) -> PixelValue {
        match self {
            ImageValue::Rgb8U(image) => PixelValue::Rgb8U(image.get(x, y)),
            ImageValue::Rgba8U(image) => PixelValue::Rgba8U(image.get(x, y)),
            ImageValue::BC1(image) => PixelValue::BC1(image.get(x, y)),
        }
    }

    fn set(&mut self, x: usize, y: usize, pixel: PixelValue) {
        match (self, pixel) {
            (ImageValue::Rgb8U(image), PixelValue::Rgb8U(pixel)) => image.set(x, y, pixel),
            (ImageValue::Rgba8U(image), PixelValue::Rgba8U(pixel)) => image.set(x, y, pixel),
            (ImageValue::BC1(image), PixelValue::BC1(pixel)) => image.set(x, y, pixel),
            (_, _) => panic!("Wrong pixel type"),
        }
    }
}

#[derive(Clone)]
enum JackalValue {
    Null,
    Uint(usize),
    Pixel(PixelValue),
    Image(ImageValue),
}

impl JackalValue {
    fn ty(&self) -> JackalType {
        match self {
            JackalValue::Null => JackalType::Null,
            JackalValue::Uint(_) => JackalType::Uint,
            JackalValue::Pixel(pixel) => pixel.ty(),
            JackalValue::Image(image) => image.ty(),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
enum JackalNode {
    Dummy,
    SourceImage(SourceImageNode),
    Filter(FilterNode),
    SizeOf(SizeOfNode),
    LZPCalculator(LZPCalculatorNode),
    LZ77Calculator(LZ77CalculatorNode),
    LZ78Calculator(LZ78CalculatorNode),
    RansCalculator(RansCalculatorNode),
    LZ77RansCalculator(LZ77RansCalculatorNode),
    LZ78RansCalculator(LZ78RansCalculatorNode),
    RleRansCalculator(RleRansCalculatorNode),
    Atlas(AtlasNode),
    ReferenceMap(ReferenceMapNode),
    BlockCopyNode(BlockCopyNode),
}

impl JackalNode {
    fn title(&self) -> String {
        match self {
            JackalNode::Dummy => "Dummy Node".into(),
            JackalNode::SourceImage(node) => node.title(),
            JackalNode::Filter(node) => node.title(),
            JackalNode::SizeOf(node) => node.title(),
            JackalNode::LZPCalculator(node) => node.title(),
            JackalNode::LZ77Calculator(node) => node.title(),
            JackalNode::LZ78Calculator(node) => node.title(),
            JackalNode::RansCalculator(node) => node.title(),
            JackalNode::LZ77RansCalculator(node) => node.title(),
            JackalNode::LZ78RansCalculator(node) => node.title(),
            JackalNode::RleRansCalculator(node) => node.title(),
            JackalNode::Atlas(node) => node.title(),
            JackalNode::ReferenceMap(node) => node.title(),
            JackalNode::BlockCopyNode(node) => node.title(),
        }
    }

    fn inputs(&self) -> usize {
        match self {
            JackalNode::Dummy => 1,
            JackalNode::SourceImage(node) => node.inputs(),
            JackalNode::Filter(node) => node.inputs(),
            JackalNode::SizeOf(node) => node.inputs(),
            JackalNode::LZPCalculator(node) => node.inputs(),
            JackalNode::LZ77Calculator(node) => node.inputs(),
            JackalNode::LZ78Calculator(node) => node.inputs(),
            JackalNode::RansCalculator(node) => node.inputs(),
            JackalNode::LZ77RansCalculator(node) => node.inputs(),
            JackalNode::LZ78RansCalculator(node) => node.inputs(),
            JackalNode::RleRansCalculator(node) => node.inputs(),
            JackalNode::Atlas(node) => node.inputs(),
            JackalNode::ReferenceMap(node) => node.inputs(),
            JackalNode::BlockCopyNode(node) => node.inputs(),
        }
    }

    /// Attempt to set input type for the node.
    fn set_input_ty(&mut self, input: usize, ty: JackalType) -> bool {
        match self {
            JackalNode::Dummy => {
                assert!(input < 1);
                return ty == JackalType::Null;
            }
            JackalNode::SourceImage(node) => node.set_input_ty(input, ty),
            JackalNode::Filter(node) => node.set_input_ty(input, ty),
            JackalNode::LZPCalculator(node) => node.set_input_ty(input, ty),
            JackalNode::LZ77Calculator(node) => node.set_input_ty(input, ty),
            JackalNode::LZ78Calculator(node) => node.set_input_ty(input, ty),
            JackalNode::SizeOf(node) => node.set_input_ty(input, ty),
            JackalNode::RansCalculator(node) => node.set_input_ty(input, ty),
            JackalNode::LZ77RansCalculator(node) => node.set_input_ty(input, ty),
            JackalNode::LZ78RansCalculator(node) => node.set_input_ty(input, ty),
            JackalNode::RleRansCalculator(node) => node.set_input_ty(input, ty),
            JackalNode::Atlas(node) => node.set_input_ty(input, ty),
            JackalNode::ReferenceMap(node) => node.set_input_ty(input, ty),
            JackalNode::BlockCopyNode(node) => node.set_input_ty(input, ty),
        }
    }

    fn input_ty(&self, input: usize) -> JackalType {
        match self {
            JackalNode::Dummy => {
                assert!(input < 1);
                JackalType::Null
            }
            JackalNode::SourceImage(node) => node.input_ty(input),
            JackalNode::Filter(node) => node.input_ty(input),
            JackalNode::LZPCalculator(node) => node.input_ty(input),
            JackalNode::LZ77Calculator(node) => node.input_ty(input),
            JackalNode::LZ78Calculator(node) => node.input_ty(input),
            JackalNode::SizeOf(node) => node.input_ty(input),
            JackalNode::RansCalculator(node) => node.input_ty(input),
            JackalNode::LZ77RansCalculator(node) => node.input_ty(input),
            JackalNode::LZ78RansCalculator(node) => node.input_ty(input),
            JackalNode::RleRansCalculator(node) => node.input_ty(input),
            JackalNode::Atlas(node) => node.input_ty(input),
            JackalNode::ReferenceMap(node) => node.input_ty(input),
            JackalNode::BlockCopyNode(node) => node.input_ty(input),
        }
    }

    fn input_ui(&mut self, input: usize, ui: &mut Ui) {
        match self {
            JackalNode::Dummy => {
                assert!(input < 1);
                ui.label("Dummy");
            }
            JackalNode::SourceImage(node) => node.input_ui(input, ui),
            JackalNode::Filter(node) => node.input_ui(input, ui),
            JackalNode::LZPCalculator(node) => node.input_ui(input, ui),
            JackalNode::LZ77Calculator(node) => node.input_ui(input, ui),
            JackalNode::LZ78Calculator(node) => node.input_ui(input, ui),
            JackalNode::SizeOf(node) => node.input_ui(input, ui),
            JackalNode::RansCalculator(node) => node.input_ui(input, ui),
            JackalNode::LZ77RansCalculator(node) => node.input_ui(input, ui),
            JackalNode::LZ78RansCalculator(node) => node.input_ui(input, ui),
            JackalNode::RleRansCalculator(node) => node.input_ui(input, ui),
            JackalNode::Atlas(node) => node.input_ui(input, ui),
            JackalNode::ReferenceMap(node) => node.input_ui(input, ui),
            JackalNode::BlockCopyNode(node) => node.input_ui(input, ui),
        }
    }

    fn outputs(&self) -> usize {
        match self {
            JackalNode::Dummy => 1,
            JackalNode::SourceImage(node) => node.outputs(),
            JackalNode::Filter(node) => node.outputs(),
            JackalNode::LZPCalculator(node) => node.outputs(),
            JackalNode::LZ77Calculator(node) => node.outputs(),
            JackalNode::LZ78Calculator(node) => node.outputs(),
            JackalNode::SizeOf(node) => node.outputs(),
            JackalNode::RansCalculator(node) => node.outputs(),
            JackalNode::LZ77RansCalculator(node) => node.outputs(),
            JackalNode::LZ78RansCalculator(node) => node.outputs(),
            JackalNode::RleRansCalculator(node) => node.outputs(),
            JackalNode::Atlas(node) => node.outputs(),
            JackalNode::ReferenceMap(node) => node.outputs(),
            JackalNode::BlockCopyNode(node) => node.outputs(),
        }
    }

    fn output_ty(&self, output: usize) -> JackalType {
        match self {
            JackalNode::Dummy => {
                assert!(output < 1);
                JackalType::Null
            }
            JackalNode::SourceImage(node) => node.output_ty(output),
            JackalNode::Filter(node) => node.output_ty(output),
            JackalNode::LZPCalculator(node) => node.output_ty(output),
            JackalNode::LZ77Calculator(node) => node.output_ty(output),
            JackalNode::LZ78Calculator(node) => node.output_ty(output),
            JackalNode::SizeOf(node) => node.output_ty(output),
            JackalNode::RansCalculator(node) => node.output_ty(output),
            JackalNode::LZ77RansCalculator(node) => node.output_ty(output),
            JackalNode::LZ78RansCalculator(node) => node.output_ty(output),
            JackalNode::RleRansCalculator(node) => node.output_ty(output),
            JackalNode::Atlas(node) => node.output_ty(output),
            JackalNode::ReferenceMap(node) => node.output_ty(output),
            JackalNode::BlockCopyNode(node) => node.output_ty(output),
        }
    }

    fn output_ui(&mut self, output: usize, ui: &mut Ui) {
        match self {
            JackalNode::Dummy => {
                assert!(output < 1);
                ui.label("Dummy");
            }
            JackalNode::SourceImage(node) => node.output_ui(output, ui),
            JackalNode::Filter(node) => node.output_ui(output, ui),
            JackalNode::LZPCalculator(node) => node.output_ui(output, ui),
            JackalNode::LZ77Calculator(node) => node.output_ui(output, ui),
            JackalNode::LZ78Calculator(node) => node.output_ui(output, ui),
            JackalNode::SizeOf(node) => node.output_ui(output, ui),
            JackalNode::RansCalculator(node) => node.output_ui(output, ui),
            JackalNode::LZ77RansCalculator(node) => node.output_ui(output, ui),
            JackalNode::LZ78RansCalculator(node) => node.output_ui(output, ui),
            JackalNode::RleRansCalculator(node) => node.output_ui(output, ui),
            JackalNode::Atlas(node) => node.output_ui(output, ui),
            JackalNode::ReferenceMap(node) => node.output_ui(output, ui),
            JackalNode::BlockCopyNode(node) => node.output_ui(output, ui),
        }
    }

    fn has_body(&self) -> bool {
        match self {
            JackalNode::Dummy => false,
            JackalNode::SourceImage(node) => node.has_body(),
            JackalNode::Filter(node) => node.has_body(),
            JackalNode::LZPCalculator(node) => node.has_body(),
            JackalNode::LZ77Calculator(node) => node.has_body(),
            JackalNode::LZ78Calculator(node) => node.has_body(),
            JackalNode::SizeOf(node) => node.has_body(),
            JackalNode::RansCalculator(node) => node.has_body(),
            JackalNode::LZ77RansCalculator(node) => node.has_body(),
            JackalNode::LZ78RansCalculator(node) => node.has_body(),
            JackalNode::RleRansCalculator(node) => node.has_body(),
            JackalNode::Atlas(node) => node.has_body(),
            JackalNode::ReferenceMap(node) => node.has_body(),
            JackalNode::BlockCopyNode(node) => node.has_body(),
        }
    }

    fn body_ui(&mut self, ui: &mut Ui) {
        match self {
            JackalNode::Dummy => unreachable!("Dummy node has no body"),
            JackalNode::SourceImage(node) => node.body_ui(ui),
            JackalNode::Filter(node) => node.body_ui(ui),
            JackalNode::LZPCalculator(node) => node.body_ui(ui),
            JackalNode::LZ77Calculator(node) => node.body_ui(ui),
            JackalNode::LZ78Calculator(node) => node.body_ui(ui),
            JackalNode::SizeOf(node) => node.body_ui(ui),
            JackalNode::RansCalculator(node) => node.body_ui(ui),
            JackalNode::LZ77RansCalculator(node) => node.body_ui(ui),
            JackalNode::LZ78RansCalculator(node) => node.body_ui(ui),
            JackalNode::RleRansCalculator(node) => node.body_ui(ui),
            JackalNode::Atlas(node) => node.body_ui(ui),
            JackalNode::ReferenceMap(node) => node.body_ui(ui),
            JackalNode::BlockCopyNode(node) => node.body_ui(ui),
        }
    }

    fn get_output(&self, output: usize) -> JackalValue {
        match self {
            JackalNode::Dummy => {
                assert!(output < 1);
                JackalValue::Null
            }
            JackalNode::SourceImage(node) => node.get_output(output),
            JackalNode::Filter(node) => node.get_output(output),
            JackalNode::LZPCalculator(node) => node.get_output(output),
            JackalNode::LZ77Calculator(node) => node.get_output(output),
            JackalNode::LZ78Calculator(node) => node.get_output(output),
            JackalNode::SizeOf(node) => node.get_output(output),
            JackalNode::RansCalculator(node) => node.get_output(output),
            JackalNode::LZ77RansCalculator(node) => node.get_output(output),
            JackalNode::LZ78RansCalculator(node) => node.get_output(output),
            JackalNode::RleRansCalculator(node) => node.get_output(output),
            JackalNode::Atlas(node) => node.get_output(output),
            JackalNode::ReferenceMap(node) => node.get_output(output),
            JackalNode::BlockCopyNode(node) => node.get_output(output),
        }
    }

    fn set_input(&mut self, input: usize, data: JackalValue) {
        match self {
            JackalNode::Dummy => {
                assert!(input < 1);
                assert_eq!(data.ty(), JackalType::Null);
            }
            JackalNode::SourceImage(node) => node.set_input(input, data),
            JackalNode::Filter(node) => node.set_input(input, data),
            JackalNode::LZPCalculator(node) => node.set_input(input, data),
            JackalNode::LZ77Calculator(node) => node.set_input(input, data),
            JackalNode::LZ78Calculator(node) => node.set_input(input, data),
            JackalNode::SizeOf(node) => node.set_input(input, data),
            JackalNode::RansCalculator(node) => node.set_input(input, data),
            JackalNode::LZ77RansCalculator(node) => node.set_input(input, data),
            JackalNode::LZ78RansCalculator(node) => node.set_input(input, data),
            JackalNode::RleRansCalculator(node) => node.set_input(input, data),
            JackalNode::Atlas(node) => node.set_input(input, data),
            JackalNode::ReferenceMap(node) => node.set_input(input, data),
            JackalNode::BlockCopyNode(node) => node.set_input(input, data),
        }
    }

    fn prepare(&mut self, ctx: &egui::Context) {
        match self {
            JackalNode::Dummy => {}
            JackalNode::SourceImage(node) => node.prepare(ctx),
            JackalNode::Filter(node) => node.prepare(ctx),
            JackalNode::LZPCalculator(node) => node.prepare(ctx),
            JackalNode::LZ77Calculator(node) => node.prepare(ctx),
            JackalNode::LZ78Calculator(node) => node.prepare(ctx),
            JackalNode::SizeOf(node) => node.prepare(ctx),
            JackalNode::RansCalculator(node) => node.prepare(ctx),
            JackalNode::LZ77RansCalculator(node) => node.prepare(ctx),
            JackalNode::LZ78RansCalculator(node) => node.prepare(ctx),
            JackalNode::RleRansCalculator(node) => node.prepare(ctx),
            JackalNode::Atlas(node) => node.prepare(ctx),
            JackalNode::ReferenceMap(node) => node.prepare(ctx),
            JackalNode::BlockCopyNode(node) => node.prepare(ctx),
        }
    }
}

struct SourceImageNode {
    file: PathBuf,
    image: Result<ImageValue, image::ImageError>,
    body: ImageWidget,
}

impl SourceImageNode {
    fn new(file: PathBuf) -> Self {
        let image = image::open(&file).map(convert_image);

        SourceImageNode {
            file,
            image,
            body: ImageWidget::new(),
        }
    }

    fn reload(&mut self) {
        self.body.unmake_texture();

        match image::open(&self.file) {
            Ok(image) => {
                self.image = Ok(convert_image(image));
            }
            Err(e) if self.image.is_err() => {
                self.image = Err(e);
            }
            Err(_) => {}
        }
    }

    fn prepare(&mut self, ctx: &egui::Context) {
        match &self.image {
            Ok(image) => self.body.make_texture(ctx, || image.to_egui()),
            Err(_) => self.body.unmake_texture(),
        }
    }

    fn title(&self) -> String {
        format!("Image")
    }

    fn inputs(&self) -> usize {
        0
    }

    fn set_input_ty(&mut self, _input: usize, _ty: JackalType) -> bool {
        unreachable!("SourceImage node has no inputs");
    }

    fn input_ty(&self, _input: usize) -> JackalType {
        unreachable!("SourceImage node has no inputs");
    }

    fn input_ui(&mut self, _input: usize, _ui: &mut Ui) {
        unreachable!("SourceImage node has no inputs");
    }

    fn set_input(&mut self, _input: usize, _data: JackalValue) {
        unreachable!("SourceImage node has no inputs");
    }

    fn outputs(&self) -> usize {
        1
    }

    fn output_ty(&self, output: usize) -> JackalType {
        assert_eq!(output, 0);
        match &self.image {
            Ok(image) => JackalType::Image(image.pixel_ty()),
            Err(_) => JackalType::Null,
        }
    }

    fn output_ui(&mut self, output: usize, ui: &mut Ui) {
        assert_eq!(output, 0);
        match &self.image {
            Ok(image) => {
                ui.label(format!("{} img", image.pixel_name()));
            }
            Err(e) => {
                ui.colored_label(Color32::RED, format!("Error: {}", e));
            }
        }
    }

    fn get_output(&self, output: usize) -> JackalValue {
        assert_eq!(output, 0);
        match &self.image {
            Ok(image) => JackalValue::Image(image.clone()),
            Err(_) => JackalValue::Null,
        }
    }

    fn has_body(&self) -> bool {
        true
    }

    fn body_ui(&mut self, ui: &mut Ui) {
        self.body.show(ui);
    }
}

impl serde::Serialize for SourceImageNode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.file.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for SourceImageNode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let file = PathBuf::deserialize(deserializer)?;
        let image = image::open(&file).map(convert_image);
        Ok(Self {
            file,
            image,
            body: ImageWidget::new(),
        })
    }
}

struct FilterNode {
    input: Option<PixelType>,
    output: Option<ImageValue>,
    filter: Filter,
    body: ImageWidget,
}

impl FilterNode {
    fn new(filter: Filter) -> Self {
        Self {
            input: None,
            output: None,
            filter,
            body: ImageWidget::new(),
        }
    }

    fn prepare(&mut self, ctx: &egui::Context) {
        match &self.output {
            Some(output) => self.body.make_texture(ctx, || output.to_egui()),
            None => self.body.unmake_texture(),
        }
    }

    fn title(&self) -> String {
        format!("{} filter", self.filter.name())
    }

    fn inputs(&self) -> usize {
        1
    }

    fn set_input_ty(&mut self, input: usize, ty: JackalType) -> bool {
        assert_eq!(input, 0);
        match ty {
            JackalType::Null => {
                self.input = None;
                true
            }
            JackalType::Image(pixel_type) => {
                self.input = Some(pixel_type);
                true
            }
            _ => false,
        }
    }

    fn input_ty(&self, input: usize) -> JackalType {
        assert_eq!(input, 0);
        match self.input {
            None => JackalType::Null,
            Some(pixel_type) => JackalType::Image(pixel_type),
        }
    }

    fn input_ui(&mut self, input: usize, _ui: &mut Ui) {
        assert_eq!(input, 0);
        // No additional UI for input
    }

    fn set_input(&mut self, input: usize, value: JackalValue) {
        assert_eq!(input, 0);

        self.output = None;
        self.body.unmake_texture();

        let image = match value {
            JackalValue::Null => return,
            JackalValue::Image(image) if Some(image.pixel_ty()) == self.input => image,
            _ => unreachable!(),
        };

        let (xs, ys) = self.filter.step();

        let input = self.input.unwrap();
        let output = self.filter.convert_type(input);

        let output = self.output.get_or_insert(ImageValue::new(
            (image.width() + xs - 1) / xs,
            (image.height() + ys - 1) / ys,
            output,
        ));

        for x in 0..(image.width() + xs - 1) / xs {
            for y in 0..(image.height() + ys - 1) / ys {
                let x = x * xs + xs - 1;
                let y = y * ys + ys - 1;

                let mut block = [[input.default(); 4]; 4];

                for i in 0..4 {
                    for j in 0..4 {
                        if x + i >= 3
                            && x + i - 3 < image.width()
                            && y + j >= 3
                            && y + j - 3 < image.height()
                        {
                            block[j][i] = image.get(x + i - 3, y + j - 3);
                        }
                    }
                }

                let r = self.filter.filter(block);

                output.set(x / xs, y / ys, r);
            }
        }
    }

    fn outputs(&self) -> usize {
        1
    }

    fn output_ty(&self, output: usize) -> JackalType {
        assert_eq!(output, 0);
        match &self.output {
            Some(output) => output.ty(),
            None => JackalType::Null,
        }
    }

    fn output_ui(&mut self, output: usize, ui: &mut Ui) {
        assert_eq!(output, 0);
        match &self.output {
            Some(output) => {
                ui.label(format!("{} image", output.pixel_name()));
            }
            None => {
                ui.colored_label(Color32::RED, "No image");
            }
        }
    }

    fn get_output(&self, output: usize) -> JackalValue {
        assert_eq!(output, 0);
        match &self.output {
            Some(output) => JackalValue::Image(output.clone()),
            None => JackalValue::Null,
        }
    }

    fn has_body(&self) -> bool {
        true
    }

    fn body_ui(&mut self, ui: &mut Ui) {
        self.body.show(ui);
    }
}

impl serde::Serialize for FilterNode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serde::Serialize::serialize(&self.filter, serializer)
    }
}

impl<'de> serde::Deserialize<'de> for FilterNode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let filter = <Filter as serde::Deserialize<'de>>::deserialize(deserializer)?;
        Ok(FilterNode {
            input: None,
            output: None,
            filter,
            body: ImageWidget::new(),
        })
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
enum Filter {
    /// Strips alpha from pixels
    ///
    /// For example RGB from RGBA, or Luma from LumaA.
    ///
    /// Noop for pixels without alpha.
    StripAlpha,

    /// Uses Paeth algorithm to predict pixel value based on top, left and top-left pixels.
    ///
    /// Outputs residual error from prediction.
    Paeth,

    /// Subtracts G channel from R and B
    GammaG,

    /// Filter that converts RGB image to BC1 format.
    BC1,
}

impl Filter {
    fn name(&self) -> &'static str {
        match self {
            Filter::StripAlpha => "Strip Alpha",
            Filter::Paeth => "Paeth",
            Filter::GammaG => "GammaG",
            Filter::BC1 => "BC1",
        }
    }

    fn convert_type(&self, input: PixelType) -> PixelType {
        match self {
            Filter::StripAlpha => match input {
                PixelType::Rgb8U => PixelType::Rgb8U,
                PixelType::Rgba8U => PixelType::Rgb8U,
                PixelType::BC1 => PixelType::BC1,
            },
            Filter::Paeth => match input {
                PixelType::Rgb8U => PixelType::Rgb8U,
                PixelType::Rgba8U => PixelType::Rgba8U,
                PixelType::BC1 => PixelType::BC1,
            },
            Filter::GammaG => match input {
                PixelType::Rgb8U => PixelType::Rgb8U,
                PixelType::Rgba8U => PixelType::Rgba8U,
                PixelType::BC1 => PixelType::BC1,
            },
            Filter::BC1 => PixelType::BC1,
        }
    }

    fn step(&self) -> (usize, usize) {
        match self {
            Filter::StripAlpha => (1, 1),
            Filter::Paeth => (1, 1),
            Filter::GammaG => (1, 1),
            Filter::BC1 => (4, 4),
        }
    }

    fn filter(&self, b: [[PixelValue; 4]; 4]) -> PixelValue {
        match self {
            Filter::StripAlpha => match b[3][3] {
                PixelValue::Rgb8U(t) => PixelValue::Rgb8U(t),
                PixelValue::Rgba8U(t) => PixelValue::Rgb8U(t.rgb()),
                PixelValue::BC1(t) => PixelValue::BC1(t),
            },
            Filter::Paeth => match (b[3][2], b[2][3], b[2][2], b[3][3]) {
                (
                    PixelValue::Rgb8U(a),
                    PixelValue::Rgb8U(b),
                    PixelValue::Rgb8U(c),
                    PixelValue::Rgb8U(t),
                ) => PixelValue::Rgb8U(paeth_rgb(a, b, c, t)),
                (
                    PixelValue::Rgba8U(a),
                    PixelValue::Rgba8U(b),
                    PixelValue::Rgba8U(c),
                    PixelValue::Rgba8U(t),
                ) => PixelValue::Rgba8U(paeth_rgba(a, b, c, t)),
                (
                    PixelValue::BC1(a),
                    PixelValue::BC1(b),
                    PixelValue::BC1(c),
                    PixelValue::BC1(t),
                ) => PixelValue::BC1(bc1::Block {
                    color0: paeth_rgb565(a.color0, b.color0, c.color0, t.color0),
                    color1: paeth_rgb565(a.color1, b.color1, c.color1, t.color1),
                    texels: t.texels,
                }),
                _ => unreachable!(),
            },
            Filter::GammaG => match b[3][3] {
                PixelValue::Rgb8U(t) => PixelValue::Rgb8U(gamma_g_rgb(t)),
                PixelValue::Rgba8U(t) => PixelValue::Rgba8U(gamma_g_rgba(t)),
                PixelValue::BC1(t) => PixelValue::BC1(bc1::Block {
                    color0: gamma_g_rgb565(t.color0),
                    color1: gamma_g_rgb565(t.color1),
                    texels: t.texels,
                }),
            },
            Filter::BC1 => {
                let b = b.map(|row| row.map(|c| c.rgb().into_f32()));
                PixelValue::BC1(bc1::Block::encode(b))
            }
        }
    }
}

fn paeth_rgb(a: Rgb8U, b: Rgb8U, c: Rgb8U, t: Rgb8U) -> Rgb8U {
    let af = Vec3::new(a.r() as f32, a.g() as f32, a.b() as f32);
    let bf = Vec3::new(b.r() as f32, b.g() as f32, b.b() as f32);
    let cf = Vec3::new(c.r() as f32, c.g() as f32, c.b() as f32);

    let pf = af + bf - cf;

    let ad = Vec3::dot(pf - af, pf - af);
    let bd = Vec3::dot(pf - bf, pf - bf);
    let cd = Vec3::dot(pf - cf, pf - cf);

    let p = if ad <= bd && ad <= cd {
        a
    } else if bd <= cd {
        b
    } else {
        c
    };

    Rgb8U::wrapping_sub(t, p)
}

fn paeth_rgb565(a: Rgb565, b: Rgb565, c: Rgb565, t: Rgb565) -> Rgb565 {
    let af = Vec3::new(a.r() as f32, a.g() as f32, a.b() as f32);
    let bf = Vec3::new(b.r() as f32, b.g() as f32, b.b() as f32);
    let cf = Vec3::new(c.r() as f32, c.g() as f32, c.b() as f32);

    let pf = af + bf - cf;

    let ad = Vec3::dot(pf - af, pf - af);
    let bd = Vec3::dot(pf - bf, pf - bf);
    let cd = Vec3::dot(pf - cf, pf - cf);

    let p = if ad <= bd && ad <= cd {
        a
    } else if bd <= cd {
        b
    } else {
        c
    };

    Rgb565::wrapping_sub(t, p)
}

fn gamma_g_rgb(t: Rgb8U) -> Rgb8U {
    Rgb8U::new(t.r().wrapping_sub(t.g()), t.g(), t.b().wrapping_sub(t.g()))
}

fn gamma_g_rgb565(t: Rgb565) -> Rgb565 {
    Rgb565::new(
        t.r().wrapping_sub(t.g() >> 1) & 31,
        t.g(),
        t.b().wrapping_sub(t.g() >> 1) & 31,
    )
}

fn paeth_rgba(a: Rgba8U, b: Rgba8U, c: Rgba8U, t: Rgba8U) -> Rgba8U {
    // let af = Vec4::new(a.r() as f32, a.g() as f32, a.b() as f32, a.a() as f32);
    // let bf = Vec4::new(b.r() as f32, b.g() as f32, b.b() as f32, b.a() as f32);
    // let cf = Vec4::new(c.r() as f32, c.g() as f32, c.b() as f32, c.a() as f32);

    // let pf = af + bf - cf;

    // let ad = Vec4::dot(pf - af, pf - af);
    // let bd = Vec4::dot(pf - bf, pf - bf);
    // let cd = Vec4::dot(pf - cf, pf - cf);

    // let p = if ad <= bd && ad <= cd {
    //     a
    // } else if bd <= cd {
    //     b
    // } else {
    //     c
    // };

    // Rgba8U::wrapping_sub(t, p)

    paeth_rgb(a.rgb(), b.rgb(), c.rgb(), t.rgb()).with_alpha(t.a())
}

fn gamma_g_rgba(t: Rgba8U) -> Rgba8U {
    gamma_g_rgb(t.rgb()).with_alpha(t.a())
}

fn rgb8u_to_egui(rgb: Rgb8U) -> egui::Color32 {
    egui::Color32::from_rgb(rgb.r(), rgb.g(), rgb.b())
}

fn rgba8u_to_egui(rgba: Rgba8U) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(rgba.r(), rgba.g(), rgba.b(), rgba.a())
}

fn rgb32f_to_egui(rgb: Rgb32F) -> egui::Color32 {
    rgb8u_to_egui(Rgb8U::from_f32(rgb))
}

fn rgb_image_to_jkl(rgb: image::Rgb<u8>) -> Rgb8U {
    Rgb8U::new(rgb[0], rgb[1], rgb[2])
}

fn rgb565_to_egui(rgb: Rgb565) -> egui::Color32 {
    rgb8u_to_egui(rgb.into_8u())
}

fn convert_image(image: image::DynamicImage) -> ImageValue {
    match image {
        image::DynamicImage::ImageRgb8(rgb_image) => ImageValue::Rgb8U(Image {
            width: rgb_image.width() as usize,
            height: rgb_image.height() as usize,
            pixels: rgb_image.pixels().map(|p| rgb_image_to_jkl(*p)).collect(),
        }),
        image::DynamicImage::ImageRgba8(rgba_image) => ImageValue::Rgba8U(Image {
            width: rgba_image.width() as usize,
            height: rgba_image.height() as usize,
            pixels: rgba_image
                .pixels()
                .map(|p| Rgba8U::new(p[0], p[1], p[2], p[3]))
                .collect(),
        }),
        image => unimplemented!("Unsupported image format: {:?}", image.color()),
    }
}

#[derive(Clone)]
struct Image<T> {
    width: usize,
    height: usize,
    pixels: Vec<T>,
}

impl<T> Image<T>
where
    T: Copy,
{
    fn as_ref(&self) -> ImageRef<'_, T> {
        ImageRef::new(self.width, self.height, &self.pixels)
    }

    fn as_mut(&mut self) -> ImageMut<'_, T> {
        ImageMut::new(self.width, self.height, &mut self.pixels)
    }

    fn solid(width: usize, height: usize, fill: T) -> Self {
        Image {
            width,
            height,
            pixels: vec![fill; (width * height)],
        }
    }

    fn new(width: usize, height: usize, pixels: Vec<T>) -> Self {
        Image {
            width,
            height,
            pixels,
        }
    }

    fn get(&self, x: usize, y: usize) -> T {
        self.pixels[y * self.width + x]
    }

    fn set(&mut self, x: usize, y: usize, value: T) {
        self.pixels[y * self.width + x] = value;
    }
}

impl Image<Rgb8U> {
    fn to_egui(&self) -> egui::ColorImage {
        egui::ColorImage {
            size: [self.width, self.height],
            source_size: egui::Vec2::new(self.width as f32, self.height as f32),
            pixels: self.pixels.iter().copied().map(rgb8u_to_egui).collect(),
        }
    }
}

impl Image<Rgba8U> {
    fn to_egui(&self) -> egui::ColorImage {
        egui::ColorImage {
            size: [self.width, self.height],
            source_size: egui::Vec2::new(self.width as f32, self.height as f32),
            pixels: self.pixels.iter().copied().map(rgba8u_to_egui).collect(),
        }
    }
}

impl Image<bc1::Block> {
    fn to_egui(&self) -> egui::ColorImage {
        egui::ColorImage {
            size: [self.width * 4, self.height * 4],
            source_size: egui::Vec2::new(self.width as f32 * 4.0, self.height as f32 * 4.0),
            pixels: {
                let mut pixels = vec![egui::Color32::BLACK; (self.width * self.height * 16)];

                for y in 0..self.height {
                    for x in 0..self.width {
                        let block = self.get(x, y).decode();
                        for j in 0..4 {
                            for i in 0..4 {
                                let pixel = block[j][i];
                                pixels[((y * 4 + j) * self.width + x) * 4 + i] =
                                    rgb32f_to_egui(pixel);
                            }
                        }
                    }
                }

                pixels
            },
        }
    }
}

struct ImageWidget {
    texture: Option<TextureHandle>,
    max_size: Vec2,
}

impl ImageWidget {
    fn new() -> Self {
        ImageWidget {
            texture: None,
            max_size: Vec2::INFINITY,
        }
    }

    fn make_texture(&mut self, ctx: &egui::Context, image: impl FnOnce() -> egui::ColorImage) {
        if self.texture.is_some() {
            return;
        }

        let texture = ctx.load_texture("image", image(), TextureOptions::NEAREST);
        self.texture = Some(texture);
    }

    fn unmake_texture(&mut self) {
        self.texture = None;
    }

    fn show(&self, ui: &mut Ui) {
        if let Some(texture) = &self.texture {
            egui::Resize::default()
                .default_size(texture.size_vec2())
                .show(ui, |ui| {
                    let original = texture.size_vec2();
                    let available = ui.available_size();

                    let x = original.x / available.x;
                    let y = original.y / available.y;

                    let size = original / x.max(y);

                    ui.image(SizedTexture {
                        size,
                        id: texture.id(),
                    });
                });
        }
    }
}

/// Calculates output size of LZP compression.
struct LZPCalculatorNode {
    input: Option<PixelType>,
    window_size: usize,
    cache_size_pow: usize,
    lzp_size: usize,
}

impl LZPCalculatorNode {
    fn new() -> Self {
        LZPCalculatorNode {
            input: None,
            window_size: 1,
            cache_size_pow: 1,
            lzp_size: 0,
        }
    }

    fn prepare(&mut self, _ctx: &egui::Context) {}

    fn title(&self) -> String {
        "LZP calculator".to_owned()
    }

    fn inputs(&self) -> usize {
        3
    }

    fn set_input_ty(&mut self, input: usize, ty: JackalType) -> bool {
        match input {
            0 => match ty {
                JackalType::Null => {
                    self.input = None;
                    true
                }
                JackalType::Image(pixel_type) => {
                    self.input = Some(pixel_type);
                    true
                }
                _ => false,
            },
            1 | 2 => matches!(ty, JackalType::Uint),
            _ => unreachable!(),
        }
    }

    fn input_ty(&self, input: usize) -> JackalType {
        match input {
            0 => match self.input {
                None => JackalType::Null,
                Some(pixel_type) => JackalType::Image(pixel_type),
            },
            1 | 2 => JackalType::Uint,
            _ => unreachable!(),
        }
    }

    fn input_ui(&mut self, input: usize, ui: &mut Ui) {
        match input {
            0 => {}
            1 => {
                ui.add(
                    egui::DragValue::new(&mut self.window_size)
                        .range(1..=10)
                        .clamp_existing_to_range(true),
                );
            }
            2 => {
                ui.add(
                    egui::DragValue::new(&mut self.cache_size_pow)
                        .range(1..=10)
                        .clamp_existing_to_range(true),
                );
            }
            _ => unreachable!(),
        }
    }

    fn set_input(&mut self, input: usize, value: JackalValue) {
        match input {
            0 => {
                self.lzp_size = 0;

                let image = match value {
                    JackalValue::Null => return,
                    JackalValue::Image(image) => {
                        assert_eq!(Some(image.pixel_ty()), self.input);
                        image
                    }
                    _ => unreachable!(),
                };

                let input = self.input.unwrap();

                let mut window = vec![input.default(); self.window_size];

                let cache_size = 1 << (self.cache_size_pow);
                let mut cache = vec![input.default(); cache_size];

                let mut total_bits = 0;

                for x in 0..image.width() {
                    for y in 0..image.height() {
                        let t = image.get(x, y);

                        let h = window.iter().fold(1, |acc, c| acc * 17 + c.hash()) as usize;
                        let p = cache[(h) % cache_size];
                        cache[(h) % cache_size] = t;

                        if p == t {
                            total_bits += 1;
                        } else {
                            total_bits += 1;
                            total_bits += input.bit_size();
                        }

                        window.rotate_left(1);
                        *window.last_mut().unwrap() = t;
                    }
                }

                self.lzp_size = total_bits;
            }
            1 => match value {
                JackalValue::Uint(value) => self.window_size = value as usize,
                _ => unreachable!(),
            },
            2 => match value {
                JackalValue::Uint(value) => self.cache_size_pow = value as usize,
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }
    }

    fn outputs(&self) -> usize {
        1
    }

    fn output_ty(&self, output: usize) -> JackalType {
        assert_eq!(output, 0);
        JackalType::Uint
    }

    fn output_ui(&mut self, output: usize, ui: &mut Ui) {
        assert_eq!(output, 0);

        let human_readable = human_readable_bits(self.lzp_size);
        let r = ui.label(human_readable);
        r.on_hover_text(format!("{} bit", self.lzp_size));
    }

    fn get_output(&self, output: usize) -> JackalValue {
        assert_eq!(output, 0);
        match &self.input {
            Some(_) => JackalValue::Uint(self.lzp_size),
            None => JackalValue::Null,
        }
    }

    fn has_body(&self) -> bool {
        false
    }

    fn body_ui(&mut self, _ui: &mut Ui) {
        unreachable!()
    }
}

impl serde::Serialize for LZPCalculatorNode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_unit()
    }
}

impl<'de> serde::Deserialize<'de> for LZPCalculatorNode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        <() as serde::Deserialize<'de>>::deserialize(deserializer)?;
        Ok(LZPCalculatorNode::new())
    }
}

/// Calculates output size of LZP compression.
struct SizeOfNode {
    input: JackalType,
    size: usize,
}

impl SizeOfNode {
    fn new() -> Self {
        SizeOfNode {
            input: JackalType::Null,
            size: 0,
        }
    }

    fn prepare(&mut self, _ctx: &egui::Context) {}

    fn title(&self) -> String {
        "SizeOf".to_owned()
    }

    fn inputs(&self) -> usize {
        1
    }

    fn set_input_ty(&mut self, input: usize, ty: JackalType) -> bool {
        assert_eq!(input, 0);
        self.input = ty;
        true
    }

    fn input_ty(&self, input: usize) -> JackalType {
        assert_eq!(input, 0);
        self.input
    }

    fn input_ui(&mut self, input: usize, _ui: &mut Ui) {
        assert_eq!(input, 0);
        // No additional UI for input
    }

    fn set_input(&mut self, input: usize, value: JackalValue) {
        assert_eq!(input, 0);

        self.size = 0;

        match value {
            JackalValue::Null => {
                assert_eq!(self.input, JackalType::Null);
                self.size = 0;
            }
            JackalValue::Uint(_) => {
                assert_eq!(self.input, JackalType::Uint);
                self.size = 64;
            }
            JackalValue::Pixel(pixel) => {
                assert_eq!(self.input, JackalType::Pixel(pixel.pixel_ty()));
                self.size = pixel.pixel_ty().bit_size();
            }
            JackalValue::Image(image) => {
                assert_eq!(self.input, JackalType::Image(image.pixel_ty()));
                self.size = image.pixel_ty().bit_size()
                    * (image.width() as usize)
                    * (image.height() as usize);
            }
        }
    }

    fn outputs(&self) -> usize {
        1
    }

    fn output_ty(&self, output: usize) -> JackalType {
        assert_eq!(output, 0);
        JackalType::Uint
    }

    fn output_ui(&mut self, output: usize, ui: &mut Ui) {
        assert_eq!(output, 0);

        let human_readable = human_readable_bits(self.size);
        let r = ui.label(human_readable);
        r.on_hover_text(format!("{} bit", self.size));
    }

    fn get_output(&self, output: usize) -> JackalValue {
        assert_eq!(output, 0);
        JackalValue::Uint(self.size)
    }

    fn has_body(&self) -> bool {
        false
    }

    fn body_ui(&mut self, _ui: &mut Ui) {
        unreachable!()
    }
}

impl serde::Serialize for SizeOfNode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_unit()
    }
}

impl<'de> serde::Deserialize<'de> for SizeOfNode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        <() as serde::Deserialize<'de>>::deserialize(deserializer)?;
        Ok(SizeOfNode::new())
    }
}

fn human_readable_bits(bits: usize) -> String {
    match bits {
        ..0x1000 => format!("{} bit", bits),
        ..0x400000 => format!("{} Kbit", bits / 0x400),
        ..0x100000000 => format!("{} Mbit", bits / 0x100000),
        ..0x40000000000 => format!("{} Gbit", bits / 0x40000000),
        _ => format!("{} Tbits", bits / 0x10000000000),
    }
}

/// Dummy io::Write type that only remembers total size.
struct WriteSize {
    size: usize,
}

impl WriteSize {
    pub fn new() -> Self {
        WriteSize { size: 0 }
    }
}

impl io::Write for WriteSize {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.size += buf.len();
        Ok(buf.len())
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.size += buf.len();
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Calculates output size of LZP compression.
struct LZ77CalculatorNode {
    input: Option<PixelType>,
    image: Option<ImageValue>,
    window_size: u32,
    sizes: [usize; 2],
    unzip_block: bool,
}

impl LZ77CalculatorNode {
    fn new() -> Self {
        LZ77CalculatorNode {
            input: None,
            image: None,
            window_size: 14,
            sizes: [0; 2],
            unzip_block: false,
        }
    }

    fn prepare(&mut self, _ctx: &egui::Context) {}

    fn title(&self) -> String {
        "LZ77 calculator".to_owned()
    }

    fn inputs(&self) -> usize {
        1
    }

    fn set_input_ty(&mut self, input: usize, ty: JackalType) -> bool {
        match input {
            0 => match ty {
                JackalType::Null => {
                    self.input = None;
                    true
                }
                JackalType::Image(pixel_type) => {
                    self.input = Some(pixel_type);
                    true
                }
                _ => false,
            },
            _ => unreachable!(),
        }
    }

    fn input_ty(&self, input: usize) -> JackalType {
        match input {
            0 => match self.input {
                None => JackalType::Null,
                Some(pixel_type) => JackalType::Image(pixel_type),
            },
            1 | 2 => JackalType::Uint,
            _ => unreachable!(),
        }
    }

    fn input_ui(&mut self, input: usize, ui: &mut Ui) {
        match input {
            0 => {}
            _ => unreachable!(),
        }
    }

    fn rebuild(&mut self) {
        self.sizes[0] = 0;
        self.sizes[1] = 0;

        let Some(image) = &self.image else {
            return;
        };

        match image.pixel_ty() {
            PixelType::Rgb8U => {
                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);

                let mut ebs = ExtendLZ77 {
                    write_bits: &mut write_bits,
                    reference_count: 0,
                    literal_count: 0,
                };

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        let mut encoder = lz77::Encoder::new(Rgb8U::BLACK, 1 << self.window_size);

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::Rgb8U(rgb) => {
                                        encoder.encode(rgb, &mut ebs);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }

                println!(
                    "references: {}, literals: {}",
                    ebs.reference_count, ebs.literal_count
                );

                write_bits.finish().unwrap();
                self.sizes[0] = write_size.size as usize * 8;
            }
            PixelType::Rgba8U => {
                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);

                let mut ebs = ExtendLZ77 {
                    write_bits: &mut write_bits,
                    reference_count: 0,
                    literal_count: 0,
                };

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        let mut encoder =
                            lz77::Encoder::<Rgba8U>::new(Rgba8U::BLACK, 1 << self.window_size);

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::Rgba8U(rgba) => {
                                        encoder.encode(rgba, &mut ebs);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }

                println!(
                    "references: {}, literals: {}",
                    ebs.reference_count, ebs.literal_count
                );

                write_bits.finish().unwrap();
                self.sizes[0] = write_size.size as usize * 8;
            }
            PixelType::BC1 if self.unzip_block => {
                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);

                let mut ebs = ExtendLZ77 {
                    write_bits: &mut write_bits,
                    reference_count: 0,
                    literal_count: 0,
                };

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        let mut encoder =
                            lz77::Encoder::<Rgb565>::new(Rgb565::BLACK, 1 << self.window_size);

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::BC1(block) => {
                                        encoder.encode(block.color0, &mut ebs);
                                        encoder.encode(block.color1, &mut ebs);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }

                println!(
                    "COLOR: references: {}, literals: {}",
                    ebs.reference_count, ebs.literal_count
                );

                write_bits.finish().unwrap();
                self.sizes[0] = write_size.size as usize * 8;

                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);

                let mut ebs = ExtendLZ77 {
                    write_bits: &mut write_bits,
                    reference_count: 0,
                    literal_count: 0,
                };

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        let mut encoder = lz77::Encoder::<[u8; 4]>::new([0u8; 4], self.window_size);

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::BC1(block) => {
                                        encoder.encode(block.texels, &mut ebs);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }

                println!(
                    "TEXEL: references: {}, literals: {}",
                    ebs.reference_count, ebs.literal_count
                );

                write_bits.finish().unwrap();
                self.sizes[1] = write_size.size as usize * 8;
            }
            PixelType::BC1 => {
                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);

                let mut ebs = ExtendLZ77 {
                    write_bits: &mut write_bits,
                    reference_count: 0,
                    literal_count: 0,
                };

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        let mut encoder = lz77::Encoder::<bc1::Block>::new(
                            bc1::Block::BLACK,
                            1 << self.window_size,
                        );

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::BC1(block) => {
                                        encoder.encode(block, &mut ebs);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }

                println!(
                    "references: {}, literals: {}",
                    ebs.reference_count, ebs.literal_count
                );

                write_bits.finish().unwrap();
                self.sizes[0] = write_size.size as usize * 8;
            }
        }
    }

    fn set_input(&mut self, input: usize, value: JackalValue) {
        match input {
            0 => {
                match value {
                    JackalValue::Null => self.image = None,
                    JackalValue::Image(image) => {
                        assert_eq!(Some(image.pixel_ty()), self.input);
                        self.image = Some(image);
                        self.rebuild();
                    }
                    _ => unreachable!(),
                };
            }
            _ => unreachable!(),
        }
    }

    fn outputs(&self) -> usize {
        1
    }

    fn output_ty(&self, output: usize) -> JackalType {
        assert_eq!(output, 0);
        JackalType::Uint
    }

    fn output_ui(&mut self, output: usize, ui: &mut Ui) {
        assert_eq!(output, 0);

        let total = self.sizes[0] + self.sizes[1];
        let human_readable = human_readable_bits(total);
        let r = ui.label(human_readable);
        r.on_hover_ui(|ui| match self.input {
            None => {
                ui.label("No input");
            }
            Some(PixelType::Rgb8U) => {
                ui.label(format!("{} bit", self.sizes[0]));
            }
            Some(PixelType::Rgba8U) => {
                ui.label(format!("MAP: {} bit", self.sizes[0]));
            }
            Some(PixelType::BC1) => {
                if self.unzip_block {
                    ui.label(format!("color: {} bit", self.sizes[0]));
                    ui.label(format!("texel: {} bit", self.sizes[1]));
                } else {
                    ui.label(format!("{} bit", self.sizes[0]));
                }
            }
        });
    }

    fn get_output(&self, output: usize) -> JackalValue {
        assert_eq!(output, 0);
        let total = self.sizes[0] + self.sizes[1];
        match &self.input {
            Some(_) => JackalValue::Uint(total),
            None => JackalValue::Null,
        }
    }

    fn has_body(&self) -> bool {
        true
    }

    fn body_ui(&mut self, ui: &mut Ui) {
        let r = ui.add(
            egui::DragValue::new(&mut self.window_size)
                .range(1..=16)
                .clamp_existing_to_range(true),
        );

        if (r.changed() && !r.dragged()) || r.drag_stopped() {
            self.rebuild();
        }

        if matches!(self.input, Some(PixelType::BC1)) {
            let r = ui.checkbox(&mut self.unzip_block, "Unzip block");
            if r.changed() {
                self.rebuild();
            }
        }
    }
}

impl serde::Serialize for LZ77CalculatorNode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_unit()
    }
}

impl<'de> serde::Deserialize<'de> for LZ77CalculatorNode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        <() as serde::Deserialize<'de>>::deserialize(deserializer)?;
        Ok(LZ77CalculatorNode::new())
    }
}

/// Calculates output size of LZP compression.
struct LZ78CalculatorNode {
    input: Option<PixelType>,
    image: Option<ImageValue>,
    sizes: [usize; 2],
    unzip_block: bool,
}

impl LZ78CalculatorNode {
    fn new() -> Self {
        LZ78CalculatorNode {
            input: None,
            image: None,
            sizes: [0; 2],
            unzip_block: false,
        }
    }

    fn prepare(&mut self, _ctx: &egui::Context) {}

    fn title(&self) -> String {
        "LZ78 calculator".to_owned()
    }

    fn inputs(&self) -> usize {
        1
    }

    fn set_input_ty(&mut self, input: usize, ty: JackalType) -> bool {
        match input {
            0 => match ty {
                JackalType::Null => {
                    self.input = None;
                    true
                }
                JackalType::Image(pixel_type) => {
                    self.input = Some(pixel_type);
                    true
                }
                _ => false,
            },
            _ => unreachable!(),
        }
    }

    fn input_ty(&self, input: usize) -> JackalType {
        match input {
            0 => match self.input {
                None => JackalType::Null,
                Some(pixel_type) => JackalType::Image(pixel_type),
            },
            1 | 2 => JackalType::Uint,
            _ => unreachable!(),
        }
    }

    fn input_ui(&mut self, input: usize, ui: &mut Ui) {
        match input {
            0 => {}
            _ => unreachable!(),
        }
    }

    fn rebuild(&mut self) {
        self.sizes[0] = 0;
        self.sizes[1] = 0;

        let Some(image) = &self.image else {
            return;
        };

        match image.pixel_ty() {
            PixelType::Rgb8U => {
                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);

                let mut ebs = ExtendLZ78 {
                    write_bits: &mut write_bits,
                };

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        let mut encoder = jkl::lz78::Encoder::new();

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::Rgb8U(rgb) => {
                                        ebs.extend(encoder.encode(rgb));
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }

                write_bits.finish().unwrap();
                self.sizes[0] = write_size.size as usize * 8;
            }
            PixelType::Rgba8U => {
                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);

                let mut ebs = ExtendLZ78 {
                    write_bits: &mut write_bits,
                };

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        let mut encoder = jkl::lz78::Encoder::<Rgba8U>::new();

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::Rgba8U(rgba) => {
                                        ebs.extend(encoder.encode(rgba));
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }

                write_bits.finish().unwrap();
                self.sizes[0] = write_size.size as usize * 8;
            }
            PixelType::BC1 if self.unzip_block => {
                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);

                let mut ebs = ExtendLZ78 {
                    write_bits: &mut write_bits,
                };

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        let mut encoder = jkl::lz78::Encoder::<Rgb565>::new();

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::BC1(block) => {
                                        ebs.extend(encoder.encode(block.color0));
                                        ebs.extend(encoder.encode(block.color1));
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }

                write_bits.finish().unwrap();
                self.sizes[0] = write_size.size as usize * 8;

                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);

                let mut ebs = ExtendLZ78 {
                    write_bits: &mut write_bits,
                };

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        let mut encoder = jkl::lz78::Encoder::<[u8; 4]>::new();

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::BC1(block) => {
                                        ebs.extend(encoder.encode(block.texels));
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }

                write_bits.finish().unwrap();
                self.sizes[1] = write_size.size as usize * 8;
            }
            PixelType::BC1 => {
                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);

                let mut ebs = ExtendLZ78 {
                    write_bits: &mut write_bits,
                };

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        let mut encoder = jkl::lz78::Encoder::<bc1::Block>::new();

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::BC1(block) => {
                                        ebs.extend(encoder.encode(block));
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }

                write_bits.finish().unwrap();
                self.sizes[0] = write_size.size as usize * 8;
            }
        }
    }

    fn set_input(&mut self, input: usize, value: JackalValue) {
        match input {
            0 => {
                match value {
                    JackalValue::Null => self.image = None,
                    JackalValue::Image(image) => {
                        assert_eq!(Some(image.pixel_ty()), self.input);
                        self.image = Some(image);
                        self.rebuild();
                    }
                    _ => unreachable!(),
                };
            }
            _ => unreachable!(),
        }
    }

    fn outputs(&self) -> usize {
        1
    }

    fn output_ty(&self, output: usize) -> JackalType {
        assert_eq!(output, 0);
        JackalType::Uint
    }

    fn output_ui(&mut self, output: usize, ui: &mut Ui) {
        assert_eq!(output, 0);

        let total = self.sizes[0] + self.sizes[1];
        let human_readable = human_readable_bits(total);
        let r = ui.label(human_readable);
        r.on_hover_ui(|ui| match self.input {
            None => {
                ui.label("No input");
            }
            Some(PixelType::Rgb8U) => {
                ui.label(format!("{} bit", self.sizes[0]));
            }
            Some(PixelType::Rgba8U) => {
                ui.label(format!("MAP: {} bit", self.sizes[0]));
            }
            Some(PixelType::BC1) => {
                if self.unzip_block {
                    ui.label(format!("color: {} bit", self.sizes[0]));
                    ui.label(format!("texel: {} bit", self.sizes[1]));
                } else {
                    ui.label(format!("{} bit", self.sizes[0]));
                }
            }
        });
    }

    fn get_output(&self, output: usize) -> JackalValue {
        assert_eq!(output, 0);
        let total = self.sizes[0] + self.sizes[1];
        match &self.input {
            Some(_) => JackalValue::Uint(total),
            None => JackalValue::Null,
        }
    }

    fn has_body(&self) -> bool {
        matches!(self.input, Some(PixelType::BC1))
    }

    fn body_ui(&mut self, ui: &mut Ui) {
        if matches!(self.input, Some(PixelType::BC1)) {
            let r = ui.checkbox(&mut self.unzip_block, "Unzip block");
            if r.changed() {
                self.rebuild();
            }
        }
    }
}

impl serde::Serialize for LZ78CalculatorNode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_unit()
    }
}

impl<'de> serde::Deserialize<'de> for LZ78CalculatorNode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        <() as serde::Deserialize<'de>>::deserialize(deserializer)?;
        Ok(LZ78CalculatorNode::new())
    }
}

/// Calculates output size of LZP compression.
struct RansCalculatorNode {
    input: Option<PixelType>,
    image: Option<ImageValue>,
    map_sizes: [usize; 2],
    rans_sizes: [usize; 2],
    unzip_block: bool,
}

impl RansCalculatorNode {
    fn new() -> Self {
        RansCalculatorNode {
            input: None,
            image: None,
            map_sizes: [0, 0],
            rans_sizes: [0, 0],
            unzip_block: false,
        }
    }

    fn prepare(&mut self, _ctx: &egui::Context) {}

    fn title(&self) -> String {
        "rANS calculator".to_owned()
    }

    fn inputs(&self) -> usize {
        1
    }

    fn set_input_ty(&mut self, input: usize, ty: JackalType) -> bool {
        match input {
            0 => match ty {
                JackalType::Null => {
                    self.input = None;
                    true
                }
                JackalType::Image(pixel_type) => {
                    self.input = Some(pixel_type);
                    true
                }
                _ => false,
            },
            _ => unreachable!(),
        }
    }

    fn input_ty(&self, input: usize) -> JackalType {
        match input {
            0 => match self.input {
                None => JackalType::Null,
                Some(pixel_type) => JackalType::Image(pixel_type),
            },
            1 | 2 => JackalType::Uint,
            _ => unreachable!(),
        }
    }

    fn input_ui(&mut self, input: usize, ui: &mut Ui) {
        match input {
            0 => {}
            _ => unreachable!(),
        }
    }

    fn rebuild(&mut self) {
        self.map_sizes[0] = 0;
        self.map_sizes[1] = 0;
        self.rans_sizes[0] = 0;
        self.rans_sizes[1] = 0;

        let Some(image) = &self.image else {
            return;
        };

        match image.pixel_ty() {
            PixelType::Rgb8U => {
                let data = (0..image.width()).flat_map(|x| {
                    let image = &image;
                    (0..image.height()).map(move |y| match image.get(x, y) {
                        PixelValue::Rgb8U(c) => c,
                        _ => unreachable!(),
                    })
                });

                let ctx = jkl::ans::Context::from_input_ord_by(data, rgb8u_ord);

                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);
                ctx.write_with_delta(&mut write_bits, Rgb8U::BLACK, rgb8u_ord, rgb8u_delta)
                    .unwrap();
                write_bits.finish().unwrap();
                self.map_sizes[0] += write_size.size as usize * 8;

                println!(
                    "rANS MAP {} -> {}",
                    ctx.freqs().len(),
                    human_readable_bits(write_size.size as usize * 8)
                );

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        let data = (x..usize::min(x + 256, image.width())).flat_map(|dx| {
                            let image = &image;
                            (y..usize::min(y + 256, image.height())).map(move |dy| {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::Rgb8U(c) => c,
                                    _ => unreachable!(),
                                }
                            })
                        });

                        let rev_data = data.rev();

                        let mut encoder = jkl::ans::Encoder::new(&ctx);

                        let mut emitted = 0;
                        rev_data.for_each(|p| {
                            if let Some(_) = encoder.encode(p) {
                                emitted += 32;
                            }
                        });

                        println!("rANS emitted: {}", human_readable_bits(emitted));

                        self.rans_sizes[0] += emitted;
                        self.rans_sizes[0] += 64;
                    }
                }
            }

            PixelType::Rgba8U => {
                let ctx = jkl::ans::Context::from_input_ord_by(
                    (0..image.width()).flat_map(|x| {
                        let image = &image;
                        (0..image.height()).map(move |y| match image.get(x, y) {
                            PixelValue::Rgba8U(c) => c,
                            _ => unreachable!(),
                        })
                    }),
                    |a, b| a.bytes().cmp(&b.bytes()),
                );

                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);
                ctx.write_with_delta(
                    &mut write_bits,
                    Rgba8U::TRANSPARENT,
                    rgba8u_ord,
                    rgba8u_delta,
                )
                .unwrap();
                write_bits.finish().unwrap();
                self.map_sizes[0] += write_size.size as usize * 8;

                println!(
                    "rANS MAP {} -> {}",
                    ctx.freqs().len(),
                    human_readable_bits(write_size.size as usize * 8)
                );

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        let data = (x..usize::min(x + 256, image.width())).flat_map(|dx| {
                            let image = &image;
                            (y..usize::min(y + 256, image.height())).map(move |dy| {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::Rgba8U(c) => c,
                                    _ => unreachable!(),
                                }
                            })
                        });

                        let rev_data = data.rev();

                        let mut encoder = jkl::ans::Encoder::new(&ctx);

                        let mut emitted = 0;
                        rev_data.for_each(|p| {
                            if let Some(_) = encoder.encode(p) {
                                emitted += 32;
                            }
                        });

                        println!("rANS emitted: {}", human_readable_bits(emitted));

                        self.rans_sizes[0] += emitted;
                        self.rans_sizes[0] += 64;
                    }
                }
            }

            PixelType::BC1 if self.unzip_block => {
                let color_ctx = jkl::ans::Context::from_input_ord_by(
                    (0..image.width()).flat_map(|x| {
                        let image = &image;
                        (0..image.height()).flat_map(move |y| match image.get(x, y) {
                            PixelValue::BC1(b) => [b.color0, b.color1],
                            _ => unreachable!(),
                        })
                    }),
                    rgb565_ord,
                );

                let texel_ctx = jkl::ans::Context::from_input((0..image.width()).flat_map(|x| {
                    let image = &image;
                    (0..image.height()).flat_map(move |y| match image.get(x, y) {
                        PixelValue::BC1(b) => b.texels,
                        _ => unreachable!(),
                    })
                }));

                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);
                color_ctx
                    .write_with_delta(&mut write_bits, Rgb565::BLACK, rgb565_ord, rgb565_delta)
                    .unwrap();
                write_bits.finish().unwrap();
                self.map_sizes[0] += write_size.size as usize * 8;

                println!(
                    "rANS MAP color {} -> {}",
                    color_ctx.freqs().len(),
                    human_readable_bits(write_size.size as usize * 8)
                );

                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);
                texel_ctx.write(&mut write_bits).unwrap();
                write_bits.finish().unwrap();
                self.map_sizes[1] += write_size.size as usize * 8;

                println!(
                    "rANS MAP texel {} -> {}",
                    texel_ctx.freqs().len(),
                    human_readable_bits(write_size.size as usize * 8)
                );

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        let data = (x..usize::min(x + 256, image.width())).flat_map(|dx| {
                            let image = &image;
                            (y..usize::min(y + 256, image.height())).map(move |dy| {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::BC1(c) => c,
                                    _ => unreachable!(),
                                }
                            })
                        });

                        let rev_data = data.rev();

                        let mut encoder = jkl::ans::Encoder::new(&color_ctx);

                        let mut emitted = 0;
                        rev_data.clone().for_each(|p| {
                            if let Some(_) = encoder.encode(p.color0) {
                                emitted += 32;
                            }
                            if let Some(_) = encoder.encode(p.color1) {
                                emitted += 32;
                            }
                        });

                        println!("rANS color emitted: {}", human_readable_bits(emitted));

                        self.rans_sizes[0] += emitted;
                        self.rans_sizes[0] += 64;

                        let mut encoder = jkl::ans::Encoder::new(&texel_ctx);

                        let mut emitted = 0;
                        rev_data.clone().for_each(|p| {
                            if let Some(_) = encoder.encode(p.texels[0]) {
                                emitted += 32;
                            }
                            if let Some(_) = encoder.encode(p.texels[1]) {
                                emitted += 32;
                            }
                            if let Some(_) = encoder.encode(p.texels[2]) {
                                emitted += 32;
                            }
                            if let Some(_) = encoder.encode(p.texels[3]) {
                                emitted += 32;
                            }
                        });

                        println!("rANS texel emitted: {}", human_readable_bits(emitted));

                        self.rans_sizes[1] += emitted;
                        self.rans_sizes[1] += 64;
                    }
                }
            }
            PixelType::BC1 => {
                let ctx = jkl::ans::Context::from_input_ord_by(
                    (0..image.width()).flat_map(|x| {
                        let image = &image;
                        (0..image.height()).map(move |y| match image.get(x, y) {
                            PixelValue::BC1(c) => c,
                            _ => unreachable!(),
                        })
                    }),
                    bc1_ord,
                );

                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);
                ctx.write_with_delta(&mut write_bits, bc1::Block::BLACK, bc1_ord, bc1_delta)
                    .unwrap();
                write_bits.finish().unwrap();
                self.map_sizes[0] += write_size.size as usize * 8;

                println!(
                    "rANS MAP {} -> {}",
                    ctx.freqs().len(),
                    human_readable_bits(write_size.size as usize * 8)
                );

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        let data = (x..usize::min(x + 256, image.width())).flat_map(|dx| {
                            let image = &image;
                            (y..usize::min(y + 256, image.height())).map(move |dy| {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::BC1(c) => c,
                                    _ => unreachable!(),
                                }
                            })
                        });

                        let rev_data = data.rev();

                        let mut encoder = jkl::ans::Encoder::new(&ctx);

                        let mut emitted = 0;
                        rev_data.for_each(|p| {
                            if let Some(_) = encoder.encode(p) {
                                emitted += 32;
                            }
                        });

                        println!("rANS emitted: {}", human_readable_bits(emitted));

                        self.rans_sizes[0] += emitted;
                        self.rans_sizes[0] += 64;
                    }
                }
            }
        }
    }

    fn set_input(&mut self, input: usize, value: JackalValue) {
        match input {
            0 => {
                match value {
                    JackalValue::Null => self.image = None,
                    JackalValue::Image(image) => {
                        assert_eq!(Some(image.pixel_ty()), self.input);
                        self.image = Some(image);
                        self.rebuild();
                    }
                    _ => unreachable!(),
                };
            }
            _ => unreachable!(),
        }
    }

    fn outputs(&self) -> usize {
        1
    }

    fn output_ty(&self, output: usize) -> JackalType {
        assert_eq!(output, 0);
        JackalType::Uint
    }

    fn output_ui(&mut self, output: usize, ui: &mut Ui) {
        assert_eq!(output, 0);

        let total = self.map_sizes[0] + self.map_sizes[1] + self.rans_sizes[0] + self.rans_sizes[1];
        let human_readable = human_readable_bits(total);
        let r = ui.label(human_readable);
        r.on_hover_ui(|ui| match self.input {
            None => {
                ui.label("No input");
            }
            Some(PixelType::Rgb8U) => {
                ui.label(format!("MAP: {} bit", self.map_sizes[0]));
                ui.label(format!("rANS: {} bit", self.rans_sizes[0]));
            }
            Some(PixelType::Rgba8U) => {
                ui.label(format!("MAP: {} bit", self.map_sizes[0]));
                ui.label(format!("rANS: {} bit", self.rans_sizes[0]));
            }
            Some(PixelType::BC1) => {
                if self.unzip_block {
                    ui.label(format!("MAP color: {} bit", self.map_sizes[0]));
                    ui.label(format!("rANS color: {} bit", self.rans_sizes[0]));
                    ui.label(format!("MAP texel: {} bit", self.map_sizes[1]));
                    ui.label(format!("rANS texel: {} bit", self.rans_sizes[1]));
                } else {
                    ui.label(format!("MAP: {} bit", self.map_sizes[0]));
                    ui.label(format!("rANS: {} bit", self.rans_sizes[0]));
                }
            }
        });
    }

    fn get_output(&self, output: usize) -> JackalValue {
        assert_eq!(output, 0);
        let total = self.map_sizes[0] + self.map_sizes[1] + self.rans_sizes[0] + self.rans_sizes[1];
        match &self.input {
            Some(_) => JackalValue::Uint(total),
            None => JackalValue::Null,
        }
    }

    fn has_body(&self) -> bool {
        matches!(self.input, Some(PixelType::BC1))
    }

    fn body_ui(&mut self, ui: &mut Ui) {
        if matches!(self.input, Some(PixelType::BC1)) {
            let r = ui.checkbox(&mut self.unzip_block, "Unzip block");
            if r.changed() {
                self.rebuild();
            }
        }
    }
}

impl serde::Serialize for RansCalculatorNode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_unit()
    }
}

impl<'de> serde::Deserialize<'de> for RansCalculatorNode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        <() as serde::Deserialize<'de>>::deserialize(deserializer)?;
        Ok(RansCalculatorNode::new())
    }
}

/// Calculates output size of LZP compression.
struct LZ77RansCalculatorNode {
    input: Option<PixelType>,
    image: Option<ImageValue>,
    window_size: usize,
    map_sizes: [usize; 2],
    rans_sizes: [usize; 2],
    unzip_block: bool,
}

impl LZ77RansCalculatorNode {
    fn new() -> Self {
        LZ77RansCalculatorNode {
            input: None,
            image: None,
            map_sizes: [0; 2],
            rans_sizes: [0; 2],
            window_size: 14,
            unzip_block: false,
        }
    }

    fn prepare(&mut self, _ctx: &egui::Context) {}

    fn title(&self) -> String {
        "LZ77+rANS calculator".to_owned()
    }

    fn inputs(&self) -> usize {
        1
    }

    fn set_input_ty(&mut self, input: usize, ty: JackalType) -> bool {
        match input {
            0 => match ty {
                JackalType::Null => {
                    self.input = None;
                    true
                }
                JackalType::Image(pixel_type) => {
                    self.input = Some(pixel_type);
                    true
                }
                _ => false,
            },
            _ => unreachable!(),
        }
    }

    fn input_ty(&self, input: usize) -> JackalType {
        match input {
            0 => match self.input {
                None => JackalType::Null,
                Some(pixel_type) => JackalType::Image(pixel_type),
            },
            1 | 2 => JackalType::Uint,
            _ => unreachable!(),
        }
    }

    fn input_ui(&mut self, input: usize, _ui: &mut Ui) {
        match input {
            0 => {}
            _ => unreachable!(),
        }
    }

    fn rebuild(&mut self) {
        self.map_sizes[0] = 0;
        self.map_sizes[1] = 0;
        self.rans_sizes[0] = 0;
        self.rans_sizes[1] = 0;

        let Some(image) = &self.image else {
            return;
        };

        match image.pixel_ty() {
            PixelType::Rgb8U => {
                let mut buffer = Vec::new();

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        let mut lz77 = lz77::Encoder::new(Rgb8U::BLACK, 1 << self.window_size);

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::Rgb8U(c) => {
                                        lz77.encode(c, &mut buffer);
                                    }
                                    _ => {}
                                }
                            }
                        }

                        lz77.finish(&mut buffer);
                    }
                }

                let ctx = jkl::ans::Context::from_input_ord_by(buffer.iter().copied(), lzrgb8u_ord);

                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);
                ctx.write_with_delta(
                    &mut write_bits,
                    lz77::Token::Literal {
                        symbol: Rgb8U::BLACK,
                    },
                    lzrgb8u_ord,
                    lzrgb8u_delta,
                )
                .unwrap();
                write_bits.finish().unwrap();
                self.map_sizes[0] += write_size.size as usize * 8;

                println!(
                    "rANS MAP {} -> {}",
                    ctx.freqs().len(),
                    human_readable_bits(write_size.size as usize * 8)
                );

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        buffer.clear();

                        let mut lz77 = lz77::Encoder::new(Rgb8U::BLACK, 1 << self.window_size);

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::Rgb8U(c) => {
                                        lz77.encode(c, &mut buffer);
                                    }
                                    _ => {}
                                }
                            }
                        }

                        lz77.finish(&mut buffer);

                        let mut write_size = WriteSize::new();
                        let mut write_bits = WriteBits::new(&mut write_size);
                        buffer
                            .iter()
                            .for_each(|t| t.var_write(&mut write_bits).unwrap());
                        write_bits.finish().unwrap();

                        println!(
                            "LZ77 Tokens {} -> {}",
                            buffer.len(),
                            human_readable_bits(write_size.size as usize * 8)
                        );

                        let mut rans = jkl::ans::Encoder::new(&ctx);

                        let mut emitted = 0;

                        for t in buffer.iter() {
                            if let Some(_) = rans.encode(*t) {
                                emitted += 32;
                            }
                        }

                        println!("rANS emitted: {}", human_readable_bits(emitted));

                        self.rans_sizes[0] += emitted;
                        self.rans_sizes[0] += 64;
                    }
                }
            }
            PixelType::Rgba8U => {
                let mut buffer = Vec::new();

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        let mut lz77 = lz77::Encoder::new(Rgba8U::BLACK, 1 << self.window_size);

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::Rgba8U(c) => {
                                        lz77.encode(c, &mut buffer);
                                    }
                                    _ => {}
                                }
                            }
                        }

                        lz77.finish(&mut buffer);
                    }
                }

                let ctx =
                    jkl::ans::Context::from_input_ord_by(buffer.iter().copied(), lzrgba8u_ord);

                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);
                ctx.write_with_delta(
                    &mut write_bits,
                    lz77::Token::Literal {
                        symbol: Rgba8U::TRANSPARENT,
                    },
                    lzrgba8u_ord,
                    lzrgba8u_delta,
                )
                .unwrap();
                write_bits.finish().unwrap();
                self.map_sizes[0] += write_size.size as usize * 8;

                println!(
                    "rANS MAP {} -> {}",
                    ctx.freqs().len(),
                    human_readable_bits(write_size.size as usize * 8)
                );

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        buffer.clear();

                        let mut lz77 = lz77::Encoder::new(Rgba8U::BLACK, 1 << self.window_size);

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::Rgba8U(c) => {
                                        lz77.encode(c, &mut buffer);
                                    }
                                    _ => {}
                                }
                            }
                        }

                        lz77.finish(&mut buffer);

                        let mut write_size = WriteSize::new();
                        let mut write_bits = WriteBits::new(&mut write_size);
                        buffer
                            .iter()
                            .for_each(|t| t.var_write(&mut write_bits).unwrap());
                        write_bits.finish().unwrap();

                        println!(
                            "LZ77 Tokens {} -> {}",
                            buffer.len(),
                            human_readable_bits(write_size.size as usize * 8)
                        );

                        let mut rans = jkl::ans::Encoder::new(&ctx);

                        let mut emitted = 0;

                        for t in buffer.iter() {
                            if let Some(_) = rans.encode(*t) {
                                emitted += 32;
                            }
                        }

                        println!("rANS emitted: {}", human_readable_bits(emitted));

                        self.rans_sizes[0] += emitted;
                        self.rans_sizes[0] += 64;
                    }
                }
            }
            PixelType::BC1 if self.unzip_block => {
                let mut color_buffer = Vec::new();

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        let mut color_lz77 =
                            lz77::Encoder::new(Rgb565::BLACK, 1 << self.window_size);

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::BC1(b) => {
                                        color_lz77.encode(b.color0, &mut color_buffer);
                                        color_lz77.encode(b.color1, &mut color_buffer);
                                    }
                                    _ => {}
                                }
                            }
                        }

                        color_lz77.finish(&mut color_buffer);
                    }
                }

                let color_ctx = jkl::ans::Context::from_input_ord_by(
                    color_buffer.iter().copied(),
                    lzrgb565_ord,
                );

                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);
                color_ctx
                    .write_with_delta(
                        &mut write_bits,
                        lz77::Token::Literal {
                            symbol: Rgb565::BLACK,
                        },
                        lzrgb565_ord,
                        lzrgb565_delta,
                    )
                    .unwrap();
                write_bits.finish().unwrap();
                self.map_sizes[0] += write_size.size as usize * 8;

                println!(
                    "rANS MAP color {} -> {}",
                    color_ctx.freqs().len(),
                    human_readable_bits(write_size.size as usize * 8)
                );

                let texel_ctx = jkl::ans::Context::from_input((0..image.width()).flat_map(|x| {
                    let image = &image;
                    (0..image.height()).flat_map(move |y| match image.get(x, y) {
                        PixelValue::BC1(b) => b.texels,
                        _ => unreachable!(),
                    })
                }));

                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);
                texel_ctx.write(&mut write_bits).unwrap();
                write_bits.finish().unwrap();
                self.map_sizes[1] += write_size.size as usize * 8;

                println!(
                    "rANS MAP texel {} -> {}",
                    texel_ctx.freqs().len(),
                    human_readable_bits(write_size.size as usize * 8)
                );

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        color_buffer.clear();

                        let mut color_lz77 =
                            lz77::Encoder::new(Rgb565::BLACK, 1 << self.window_size);

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::BC1(b) => {
                                        color_lz77.encode(b.color0, &mut color_buffer);
                                        color_lz77.encode(b.color1, &mut color_buffer);
                                    }
                                    _ => {}
                                }
                            }
                        }

                        color_lz77.finish(&mut color_buffer);

                        let mut write_size = WriteSize::new();
                        let mut write_bits = WriteBits::new(&mut write_size);
                        color_buffer
                            .iter()
                            .for_each(|t| t.var_write(&mut write_bits).unwrap());
                        write_bits.finish().unwrap();

                        println!(
                            "LZ77 Tokens color {} -> {}",
                            color_buffer.len(),
                            human_readable_bits(write_size.size as usize * 8)
                        );

                        let mut color_rans = jkl::ans::Encoder::new(&color_ctx);
                        let mut texel_rans = jkl::ans::Encoder::new(&texel_ctx);

                        let mut emitted = 0;

                        for c in color_buffer.iter() {
                            if let Some(_) = color_rans.encode(*c) {
                                emitted += 32;
                            }
                        }

                        self.rans_sizes[0] += emitted;
                        self.rans_sizes[0] += 64;

                        println!("rANS color emitted: {}", human_readable_bits(emitted));

                        let mut emitted = 0;

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::BC1(b) => {
                                        if let Some(_) = texel_rans.encode(b.texels[0]) {
                                            emitted += 32;
                                        }
                                        if let Some(_) = texel_rans.encode(b.texels[1]) {
                                            emitted += 32;
                                        }
                                        if let Some(_) = texel_rans.encode(b.texels[2]) {
                                            emitted += 32;
                                        }
                                        if let Some(_) = texel_rans.encode(b.texels[3]) {
                                            emitted += 32;
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }

                        self.rans_sizes[1] += emitted;
                        self.rans_sizes[1] += 64;

                        println!("rANS texel emitted: {}", human_readable_bits(emitted));
                    }
                }
            }
            PixelType::BC1 => {
                let mut buffer = Vec::new();

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        let mut lz77 = lz77::Encoder::new(bc1::Block::BLACK, 1 << self.window_size);

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::BC1(b) => {
                                        lz77.encode(b, &mut buffer);
                                    }
                                    _ => {}
                                }
                            }
                        }

                        lz77.finish(&mut buffer);
                    }
                }

                let ctx = jkl::ans::Context::from_input_ord_by(buffer.iter().copied(), lzbc1_ord);

                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);
                ctx.write_ord_by(&mut write_bits, lzbc1_ord).unwrap();
                write_bits.finish().unwrap();
                self.map_sizes[0] += write_size.size as usize * 8;

                println!(
                    "rANS MAP {} -> {}",
                    ctx.freqs().len(),
                    human_readable_bits(write_size.size as usize * 8)
                );

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        buffer.clear();

                        let mut lz77 = lz77::Encoder::new(bc1::Block::BLACK, 1 << self.window_size);

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::BC1(b) => {
                                        lz77.encode(b, &mut buffer);
                                    }
                                    _ => {}
                                }
                            }
                        }

                        lz77.finish(&mut buffer);

                        let mut write_size = WriteSize::new();
                        let mut write_bits = WriteBits::new(&mut write_size);
                        buffer
                            .iter()
                            .for_each(|t| t.var_write(&mut write_bits).unwrap());
                        write_bits.finish().unwrap();

                        println!(
                            "LZ77 Tokens {} -> {}",
                            buffer.len(),
                            human_readable_bits(write_size.size as usize * 8)
                        );

                        let mut rans = jkl::ans::Encoder::new(&ctx);

                        let mut emitted = 0;

                        for t in buffer.iter() {
                            if let Some(_) = rans.encode(*t) {
                                emitted += 32;
                            }
                        }

                        println!("rANS emitted: {}", human_readable_bits(emitted));

                        self.rans_sizes[0] += emitted;
                        self.rans_sizes[0] += 64;
                    }
                }
            }
        }
    }

    fn set_input(&mut self, input: usize, value: JackalValue) {
        match input {
            0 => {
                match value {
                    JackalValue::Null => self.image = None,
                    JackalValue::Image(image) => {
                        assert_eq!(Some(image.pixel_ty()), self.input);
                        self.image = Some(image);
                        self.rebuild();
                    }
                    _ => unreachable!(),
                };
            }
            _ => unreachable!(),
        }
    }

    fn outputs(&self) -> usize {
        1
    }

    fn output_ty(&self, output: usize) -> JackalType {
        assert_eq!(output, 0);
        JackalType::Uint
    }

    fn output_ui(&mut self, output: usize, ui: &mut Ui) {
        assert_eq!(output, 0);

        let total = self.map_sizes[0] + self.map_sizes[1] + self.rans_sizes[0] + self.rans_sizes[1];
        let human_readable = human_readable_bits(total);
        let r = ui.label(human_readable);
        r.on_hover_ui(|ui| match self.input {
            None => {
                ui.label("No input");
            }
            Some(PixelType::Rgb8U) => {
                ui.label(format!("MAP: {} bit", self.map_sizes[0]));
                ui.label(format!("rANS: {} bit", self.rans_sizes[0]));
            }
            Some(PixelType::Rgba8U) => {
                ui.label(format!("MAP: {} bit", self.map_sizes[0]));
                ui.label(format!("rANS: {} bit", self.rans_sizes[0]));
            }
            Some(PixelType::BC1) => {
                if self.unzip_block {
                    ui.label(format!("MAP color: {} bit", self.map_sizes[0]));
                    ui.label(format!("rANS color: {} bit", self.rans_sizes[0]));
                    ui.label(format!("MAP texel: {} bit", self.map_sizes[1]));
                    ui.label(format!("rANS texel: {} bit", self.rans_sizes[1]));
                } else {
                    ui.label(format!("MAP: {} bit", self.map_sizes[0]));
                    ui.label(format!("rANS: {} bit", self.rans_sizes[0]));
                }
            }
        });
    }

    fn get_output(&self, output: usize) -> JackalValue {
        assert_eq!(output, 0);
        let total = self.map_sizes[0] + self.map_sizes[1] + self.rans_sizes[0] + self.rans_sizes[1];
        match &self.input {
            Some(_) => JackalValue::Uint(total),
            None => JackalValue::Null,
        }
    }

    fn has_body(&self) -> bool {
        true
    }

    fn body_ui(&mut self, ui: &mut Ui) {
        let r = ui.add(
            egui::DragValue::new(&mut self.window_size)
                .range(1..=16)
                .clamp_existing_to_range(true),
        );

        if (r.changed() && !r.dragged()) || r.drag_stopped() {
            self.rebuild();
        }

        if matches!(self.input, Some(PixelType::BC1)) {
            let r = ui.checkbox(&mut self.unzip_block, "Unzip block");
            if r.changed() {
                self.rebuild();
            }
        }
    }
}

impl serde::Serialize for LZ77RansCalculatorNode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_unit()
    }
}

impl<'de> serde::Deserialize<'de> for LZ77RansCalculatorNode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        <() as serde::Deserialize<'de>>::deserialize(deserializer)?;
        Ok(LZ77RansCalculatorNode::new())
    }
}

/// Calculates output size of LZP compression.
struct LZ78RansCalculatorNode {
    input: Option<PixelType>,
    image: Option<ImageValue>,
    map_sizes: [usize; 2],
    rans_sizes: [usize; 2],
    unzip_block: bool,
}

impl LZ78RansCalculatorNode {
    fn new() -> Self {
        LZ78RansCalculatorNode {
            input: None,
            image: None,
            map_sizes: [0; 2],
            rans_sizes: [0; 2],
            unzip_block: false,
        }
    }

    fn prepare(&mut self, _ctx: &egui::Context) {}

    fn title(&self) -> String {
        "LZ78+rANS calculator".to_owned()
    }

    fn inputs(&self) -> usize {
        1
    }

    fn set_input_ty(&mut self, input: usize, ty: JackalType) -> bool {
        match input {
            0 => match ty {
                JackalType::Null => {
                    self.input = None;
                    true
                }
                JackalType::Image(pixel_type) => {
                    self.input = Some(pixel_type);
                    true
                }
                _ => false,
            },
            _ => unreachable!(),
        }
    }

    fn input_ty(&self, input: usize) -> JackalType {
        match input {
            0 => match self.input {
                None => JackalType::Null,
                Some(pixel_type) => JackalType::Image(pixel_type),
            },
            1 | 2 => JackalType::Uint,
            _ => unreachable!(),
        }
    }

    fn input_ui(&mut self, input: usize, _ui: &mut Ui) {
        match input {
            0 => {}
            _ => unreachable!(),
        }
    }

    fn rebuild(&mut self) {
        self.map_sizes[0] = 0;
        self.map_sizes[1] = 0;
        self.rans_sizes[0] = 0;
        self.rans_sizes[1] = 0;

        let Some(image) = &self.image else {
            return;
        };

        match image.pixel_ty() {
            PixelType::Rgb8U => {
                let mut buffer = Vec::new();

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        let mut lz78 = jkl::lz78::Encoder::new();

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::Rgb8U(c) => {
                                        buffer.extend(lz78.encode(c.bits_interleaved()));
                                    }
                                    _ => {}
                                }
                            }
                        }

                        buffer.extend(lz78.finish());
                    }
                }

                let ctx = jkl::ans::Context::from_input(buffer.iter().copied());

                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);
                ctx.write(&mut write_bits).unwrap();
                write_bits.finish().unwrap();
                self.map_sizes[0] += write_size.size as usize * 8;

                println!(
                    "rANS MAP {} -> {}",
                    ctx.freqs().len(),
                    human_readable_bits(write_size.size as usize * 8)
                );

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        buffer.clear();

                        let mut lz78 = jkl::lz78::Encoder::new();

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::Rgb8U(c) => {
                                        buffer.extend(lz78.encode(c.bits_interleaved()));
                                    }
                                    _ => {}
                                }
                            }
                        }

                        buffer.extend(lz78.finish());

                        let mut write_size = WriteSize::new();
                        let mut write_bits = WriteBits::new(&mut write_size);
                        buffer
                            .iter()
                            .for_each(|t| t.var_write(&mut write_bits).unwrap());
                        write_bits.finish().unwrap();

                        println!(
                            "LZ78 Tokens {} -> {}",
                            buffer.len(),
                            human_readable_bits(write_size.size as usize * 8)
                        );

                        let mut rans = jkl::ans::Encoder::new(&ctx);

                        let mut emitted = 0;

                        for t in buffer.iter() {
                            if let Some(_) = rans.encode(*t) {
                                emitted += 32;
                            }
                        }

                        println!("rANS emitted: {}", human_readable_bits(emitted));

                        self.rans_sizes[0] += emitted;
                        self.rans_sizes[0] += 64;
                    }
                }
            }
            PixelType::Rgba8U => {
                let mut buffer = Vec::new();

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        let mut lz78 = jkl::lz78::Encoder::new();

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::Rgba8U(c) => {
                                        buffer.extend(lz78.encode(c.bits_interleaved()));
                                    }
                                    _ => {}
                                }
                            }
                        }

                        buffer.extend(lz78.finish());
                    }
                }

                let ctx = jkl::ans::Context::from_input(buffer.iter().copied());

                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);
                ctx.write(&mut write_bits).unwrap();
                write_bits.finish().unwrap();
                self.map_sizes[0] += write_size.size as usize * 8;

                println!(
                    "rANS MAP {} -> {}",
                    ctx.freqs().len(),
                    human_readable_bits(write_size.size as usize * 8)
                );

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        buffer.clear();

                        let mut lz78 = jkl::lz78::Encoder::new();

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::Rgba8U(c) => {
                                        buffer.extend(lz78.encode(c.bits_interleaved()));
                                    }
                                    _ => {}
                                }
                            }
                        }

                        buffer.extend(lz78.finish());

                        let mut write_size = WriteSize::new();
                        let mut write_bits = WriteBits::new(&mut write_size);
                        buffer
                            .iter()
                            .for_each(|t| t.var_write(&mut write_bits).unwrap());
                        write_bits.finish().unwrap();

                        println!(
                            "LZ78 Tokens {} -> {}",
                            buffer.len(),
                            human_readable_bits(write_size.size as usize * 8)
                        );

                        let mut rans = jkl::ans::Encoder::new(&ctx);

                        let mut emitted = 0;

                        for t in buffer.iter() {
                            if let Some(_) = rans.encode(*t) {
                                emitted += 32;
                            }
                        }

                        println!("rANS emitted: {}", human_readable_bits(emitted));

                        self.rans_sizes[0] += emitted;
                        self.rans_sizes[0] += 64;
                    }
                }
            }
            PixelType::BC1 if self.unzip_block => {
                let mut color_buffer = Vec::new();

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        let mut color_lz78 = jkl::lz78::Encoder::new();

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::BC1(b) => {
                                        color_buffer
                                            .extend(color_lz78.encode(b.color0.bits_interleaved()));
                                        color_buffer
                                            .extend(color_lz78.encode(b.color1.bits_interleaved()));
                                    }
                                    _ => {}
                                }
                            }
                        }

                        color_buffer.extend(color_lz78.finish());
                    }
                }

                let color_ctx = jkl::ans::Context::from_input(color_buffer.iter().copied());

                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);
                color_ctx.write(&mut write_bits).unwrap();
                write_bits.finish().unwrap();
                self.map_sizes[0] += write_size.size as usize * 8;

                println!(
                    "rANS MAP color {} -> {}",
                    color_ctx.freqs().len(),
                    human_readable_bits(write_size.size as usize * 8)
                );

                let texel_ctx = jkl::ans::Context::from_input((0..image.width()).flat_map(|x| {
                    let image = &image;
                    (0..image.height()).map(move |y| match image.get(x, y) {
                        PixelValue::BC1(b) => b.texels,
                        _ => unreachable!(),
                    })
                }));

                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);
                texel_ctx.write(&mut write_bits).unwrap();
                write_bits.finish().unwrap();
                self.map_sizes[1] += write_size.size as usize * 8;

                println!(
                    "rANS MAP texel {} -> {}",
                    texel_ctx.freqs().len(),
                    human_readable_bits(write_size.size as usize * 8)
                );

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        color_buffer.clear();

                        let mut color_lz78 = jkl::lz78::Encoder::new();

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::BC1(b) => {
                                        color_buffer
                                            .extend(color_lz78.encode(b.color0.bits_interleaved()));
                                        color_buffer
                                            .extend(color_lz78.encode(b.color1.bits_interleaved()));
                                    }
                                    _ => {}
                                }
                            }
                        }

                        color_buffer.extend(color_lz78.finish());

                        let mut write_size = WriteSize::new();
                        let mut write_bits = WriteBits::new(&mut write_size);
                        color_buffer
                            .iter()
                            .for_each(|t| t.var_write(&mut write_bits).unwrap());
                        write_bits.finish().unwrap();

                        println!(
                            "LZ78 Tokens color {} -> {}",
                            color_buffer.len(),
                            human_readable_bits(write_size.size as usize * 8)
                        );

                        let mut color_rans = jkl::ans::Encoder::new(&color_ctx);
                        let mut texel_rans = jkl::ans::Encoder::new(&texel_ctx);

                        let mut emitted = 0;

                        for c in color_buffer.iter() {
                            if let Some(_) = color_rans.encode(*c) {
                                emitted += 32;
                            }
                        }

                        self.rans_sizes[0] += emitted;
                        self.rans_sizes[0] += 64;

                        println!("rANS color emitted: {}", human_readable_bits(emitted));

                        let mut emitted = 0;

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::BC1(b) => {
                                        if let Some(_) = texel_rans.encode(b.texels) {
                                            emitted += 32;
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }

                        self.rans_sizes[1] += emitted;
                        self.rans_sizes[1] += 64;

                        println!("rANS texel emitted: {}", human_readable_bits(emitted));
                    }
                }
            }
            PixelType::BC1 => {
                let mut buffer = Vec::new();

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        let mut lz78 = jkl::lz78::Encoder::new();

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::BC1(b) => {
                                        buffer.extend(lz78.encode(b.bytes()));
                                    }
                                    _ => {}
                                }
                            }
                        }

                        buffer.extend(lz78.finish());
                    }
                }

                let ctx = jkl::ans::Context::from_input(buffer.iter().copied());

                let mut write_size = WriteSize::new();
                let mut write_bits = WriteBits::new(&mut write_size);
                ctx.write(&mut write_bits).unwrap();
                write_bits.finish().unwrap();
                self.map_sizes[0] += write_size.size as usize * 8;

                println!(
                    "rANS MAP {} -> {}",
                    ctx.freqs().len(),
                    human_readable_bits(write_size.size as usize * 8)
                );

                for x in (0..image.width()).step_by(256) {
                    for y in (0..image.height()).step_by(256) {
                        buffer.clear();

                        let mut lz78 = jkl::lz78::Encoder::new();

                        for dx in x..usize::min(x + 256, image.width()) {
                            for dy in y..usize::min(y + 256, image.height()) {
                                let p = image.get(dx, dy);

                                match p {
                                    PixelValue::BC1(b) => {
                                        buffer.extend(lz78.encode(b.bytes()));
                                    }
                                    _ => {}
                                }
                            }
                        }

                        buffer.extend(lz78.finish());

                        let mut write_size = WriteSize::new();
                        let mut write_bits = WriteBits::new(&mut write_size);
                        buffer
                            .iter()
                            .for_each(|t| t.var_write(&mut write_bits).unwrap());
                        write_bits.finish().unwrap();

                        println!(
                            "LZ78 Tokens {} -> {}",
                            buffer.len(),
                            human_readable_bits(write_size.size as usize * 8)
                        );

                        let mut rans = jkl::ans::Encoder::new(&ctx);

                        let mut emitted = 0;

                        for t in buffer.iter() {
                            if let Some(_) = rans.encode(*t) {
                                emitted += 32;
                            }
                        }

                        println!("rANS emitted: {}", human_readable_bits(emitted));

                        self.rans_sizes[0] += emitted;
                        self.rans_sizes[0] += 64;
                    }
                }
            }
        }
    }

    fn set_input(&mut self, input: usize, value: JackalValue) {
        match input {
            0 => {
                match value {
                    JackalValue::Null => self.image = None,
                    JackalValue::Image(image) => {
                        assert_eq!(Some(image.pixel_ty()), self.input);
                        self.image = Some(image);
                        self.rebuild();
                    }
                    _ => unreachable!(),
                };
            }
            _ => unreachable!(),
        }
    }

    fn outputs(&self) -> usize {
        1
    }

    fn output_ty(&self, output: usize) -> JackalType {
        assert_eq!(output, 0);
        JackalType::Uint
    }

    fn output_ui(&mut self, output: usize, ui: &mut Ui) {
        assert_eq!(output, 0);

        let total = self.map_sizes[0] + self.map_sizes[1] + self.rans_sizes[0] + self.rans_sizes[1];
        let human_readable = human_readable_bits(total);
        let r = ui.label(human_readable);
        r.on_hover_ui(|ui| match self.input {
            None => {
                ui.label("No input");
            }
            Some(PixelType::Rgb8U) => {
                ui.label(format!("MAP: {} bit", self.map_sizes[0]));
                ui.label(format!("rANS: {} bit", self.rans_sizes[0]));
            }
            Some(PixelType::Rgba8U) => {
                ui.label(format!("MAP: {} bit", self.map_sizes[0]));
                ui.label(format!("rANS: {} bit", self.rans_sizes[0]));
            }
            Some(PixelType::BC1) => {
                if self.unzip_block {
                    ui.label(format!("MAP color: {} bit", self.map_sizes[0]));
                    ui.label(format!("rANS color: {} bit", self.rans_sizes[0]));
                    ui.label(format!("MAP texel: {} bit", self.map_sizes[1]));
                    ui.label(format!("rANS texel: {} bit", self.rans_sizes[1]));
                } else {
                    ui.label(format!("MAP: {} bit", self.map_sizes[0]));
                    ui.label(format!("rANS: {} bit", self.rans_sizes[0]));
                }
            }
        });
    }

    fn get_output(&self, output: usize) -> JackalValue {
        assert_eq!(output, 0);
        let total = self.map_sizes[0] + self.map_sizes[1] + self.rans_sizes[0] + self.rans_sizes[1];
        match &self.input {
            Some(_) => JackalValue::Uint(total),
            None => JackalValue::Null,
        }
    }

    fn has_body(&self) -> bool {
        matches!(self.input, Some(PixelType::BC1))
    }

    fn body_ui(&mut self, ui: &mut Ui) {
        if matches!(self.input, Some(PixelType::BC1)) {
            let r = ui.checkbox(&mut self.unzip_block, "Unzip block");
            if r.changed() {
                self.rebuild();
            }
        }
    }
}

impl serde::Serialize for LZ78RansCalculatorNode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_unit()
    }
}

impl<'de> serde::Deserialize<'de> for LZ78RansCalculatorNode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        <() as serde::Deserialize<'de>>::deserialize(deserializer)?;
        Ok(LZ78RansCalculatorNode::new())
    }
}

struct ExtendLZ77<'a, 'b> {
    write_bits: &'a mut WriteBits<&'b mut WriteSize>,
    reference_count: usize,
    literal_count: usize,
}

impl<T> Extend<lz77::Token<T>> for ExtendLZ77<'_, '_>
where
    T: FixedCode,
{
    fn extend<I: IntoIterator<Item = lz77::Token<T>>>(&mut self, iter: I) {
        for token in iter {
            match token {
                lz77::Token::Reference { .. } => self.reference_count += 1,
                lz77::Token::Literal { .. } => self.literal_count += 1,
            }

            token.var_write(self.write_bits).unwrap();
        }
    }
}

struct ExtendLZ78<'a, 'b> {
    write_bits: &'a mut WriteBits<&'b mut WriteSize>,
}

impl<T> Extend<jkl::lz78::Token<T>> for ExtendLZ78<'_, '_>
where
    T: FixedCode,
{
    fn extend<I: IntoIterator<Item = jkl::lz78::Token<T>>>(&mut self, iter: I) {
        for token in iter {
            token.var_write(self.write_bits).unwrap();
        }
    }
}

/// Builds texture atlas.
struct AtlasNode {
    size: (usize, usize),
    inputs: [Option<ImageValue>; 8],
    atlas: Image<Rgb8U>,
    body: ImageWidget,
}

impl AtlasNode {
    fn new() -> Self {
        AtlasNode::with_size((0, 0))
    }

    fn with_size(size: (usize, usize)) -> Self {
        AtlasNode {
            size,
            inputs: array::from_fn(|_| None),
            atlas: Image::solid(size.0, size.1, Rgb8U::BLACK),
            body: ImageWidget::new(),
        }
    }

    fn prepare(&mut self, ctx: &egui::Context) {
        self.body.make_texture(ctx, || self.atlas.to_egui())
    }

    fn title(&self) -> String {
        "Atlas".to_owned()
    }

    fn inputs(&self) -> usize {
        10
    }

    fn set_input_ty(&mut self, input: usize, ty: JackalType) -> bool {
        match input {
            0 | 1 => match ty {
                JackalType::Uint => true,
                _ => false,
            },
            2..10 => match ty {
                JackalType::Null => {
                    self.inputs[input - 2] = None;
                    true
                }
                JackalType::Image(_) => {
                    self.inputs[input - 2] = None;
                    true
                }
                _ => false,
            },
            _ => unreachable!(),
        }
    }

    fn input_ty(&self, input: usize) -> JackalType {
        match input {
            0 | 1 => JackalType::Uint,
            2..10 => match &self.inputs[input - 2] {
                None => JackalType::Null,
                Some(image) => JackalType::Image(image.pixel_ty()),
            },
            _ => unreachable!(),
        }
    }

    fn input_ui(&mut self, input: usize, ui: &mut Ui) {
        let changed = match input {
            0 => ui
                .add(
                    egui::DragValue::new(&mut self.size.0)
                        .range(1..=8192)
                        .clamp_existing_to_range(true),
                )
                .changed(),
            1 => ui
                .add(
                    egui::DragValue::new(&mut self.size.1)
                        .range(1..=8192)
                        .clamp_existing_to_range(true),
                )
                .changed(),
            2..10 => false,
            _ => unreachable!(),
        };

        if changed {
            self.rebuild();
        }
    }

    fn set_input(&mut self, input: usize, value: JackalValue) {
        match input {
            0 => match value {
                JackalValue::Uint(value) => {
                    self.size.0 = value as usize;
                }
                _ => unreachable!(),
            },
            1 => match value {
                JackalValue::Uint(value) => {
                    self.size.1 = value as usize;
                }
                _ => unreachable!(),
            },
            2..10 => match value {
                JackalValue::Null => {
                    self.inputs[input - 2] = None;
                }
                JackalValue::Image(image) => {
                    self.inputs[input - 2] = Some(image);
                }
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }

        self.rebuild();
    }

    fn outputs(&self) -> usize {
        1
    }

    fn output_ty(&self, output: usize) -> JackalType {
        assert_eq!(output, 0);
        JackalType::Image(PixelType::Rgb8U)
    }

    fn output_ui(&mut self, output: usize, ui: &mut Ui) {
        assert_eq!(output, 0);
        ui.label("Rgb8U image");
    }

    fn get_output(&self, output: usize) -> JackalValue {
        assert_eq!(output, 0);
        JackalValue::Image(ImageValue::Rgb8U(self.atlas.clone()))
    }

    fn has_body(&self) -> bool {
        true
    }

    fn body_ui(&mut self, ui: &mut Ui) {
        self.body.show(ui);
    }

    fn rebuild(&mut self) {
        self.atlas = Image::solid(self.size.0, self.size.1, Rgb8U::BLACK);

        let mut pack = MaximalRectangles::new(self.size.0, self.size.1);

        let mut order: [_; 8] = array::from_fn(|x| x);

        order.sort_by_key(|i| {
            self.inputs[*i]
                .as_ref()
                .map_or(usize::MAX, |img| !(img.height() * img.width()))
        });

        for idx in order {
            if let Some(input) = &self.inputs[idx] {
                if let Some(r) = pack.insert(input.width(), input.height()) {
                    if r.w == input.width() {
                        for y in 0..r.h {
                            for x in 0..r.w {
                                let p = input.get(x, y).rgb();
                                self.atlas.set(r.x + x, r.y + y, p);
                            }
                        }
                    } else {
                        for y in 0..r.h {
                            for x in 0..r.w {
                                let p = input.get(y, x).rgb();
                                self.atlas.set(r.x + x, r.y + y, p);
                            }
                        }
                    }
                }
            }
        }

        self.body.unmake_texture();
    }
}

impl serde::Serialize for AtlasNode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.size.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for AtlasNode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let size = <(usize, usize) as serde::Deserialize<'de>>::deserialize(deserializer)?;
        Ok(AtlasNode::with_size(size))
    }
}

//////////////////////////////////////////////////
///
/// Calculates output size of LZP compression.
struct RleRansCalculatorNode {
    only_power_of_two: bool,
    unzip: bool,
    input: Option<PixelType>,
    rans_size: usize,
    image: Option<ImageValue>,
}

impl RleRansCalculatorNode {
    fn new() -> Self {
        RleRansCalculatorNode {
            only_power_of_two: false,
            unzip: false,
            input: None,
            rans_size: 0,
            image: None,
        }
    }

    fn prepare(&mut self, _ctx: &egui::Context) {}

    fn title(&self) -> String {
        "RLE+rANS calculator".to_owned()
    }

    fn inputs(&self) -> usize {
        1
    }

    fn set_input_ty(&mut self, input: usize, ty: JackalType) -> bool {
        match input {
            0 => match ty {
                JackalType::Null => {
                    self.input = None;
                    true
                }
                JackalType::Image(pixel_type) => {
                    self.input = Some(pixel_type);
                    true
                }
                _ => false,
            },
            _ => unreachable!(),
        }
    }

    fn input_ty(&self, input: usize) -> JackalType {
        match input {
            0 => match self.input {
                None => JackalType::Null,
                Some(pixel_type) => JackalType::Image(pixel_type),
            },
            1 | 2 => JackalType::Uint,
            _ => unreachable!(),
        }
    }

    fn input_ui(&mut self, input: usize, _ui: &mut Ui) {
        match input {
            0 => {}
            _ => unreachable!(),
        }
    }

    fn set_input(&mut self, input: usize, value: JackalValue) {
        match input {
            0 => {
                self.image = match value {
                    JackalValue::Null => None,
                    JackalValue::Image(image) => {
                        assert_eq!(Some(image.pixel_ty()), self.input);
                        Some(image)
                    }
                    _ => unreachable!(),
                };

                self.rebuild();
            }
            _ => unreachable!(),
        }
    }

    fn outputs(&self) -> usize {
        1
    }

    fn output_ty(&self, output: usize) -> JackalType {
        assert_eq!(output, 0);
        JackalType::Uint
    }

    fn output_ui(&mut self, output: usize, ui: &mut Ui) {
        assert_eq!(output, 0);

        let human_readable = human_readable_bits(self.rans_size);
        let r = ui.label(human_readable);
        r.on_hover_text(format!("{} bit", self.rans_size));
    }

    fn get_output(&self, output: usize) -> JackalValue {
        assert_eq!(output, 0);
        match &self.input {
            Some(_) => JackalValue::Uint(self.rans_size),
            None => JackalValue::Null,
        }
    }

    fn has_body(&self) -> bool {
        true
    }

    fn body_ui(&mut self, ui: &mut Ui) {
        let r = ui.toggle_value(&mut self.only_power_of_two, "RLE 2^x");
        if r.changed() {
            self.rebuild();
        }

        let r = ui.toggle_value(&mut self.unzip, "Unzip");
        if r.changed() {
            self.rebuild();
        }
    }

    fn rebuild(&mut self) {
        // let Some(image) = &self.image else {
        //     return;
        // };

        // self.rans_size = 0;
        // let rle_cfg = RleCfg {
        //     only_power_of_two: self.only_power_of_two,
        //     ..Default::default()
        // };

        // if self.unzip {
        //     match self.input {
        //         None => {}
        //         Some(PixelType::Rgb8U) => {
        //             let rle_data = (0..image.width()).step_by(256).flat_map(move |x| {
        //                 (0..image.height()).step_by(256).flat_map(move |y| {
        //                     let data =
        //                         (x..usize::min(x + 256, image.width())).flat_map(move |dx| {
        //                             (y..usize::min(y + 256, image.height())).map(move |dy| {
        //                                 let p = image.get(dx, dy);

        //                                 match p {
        //                                     PixelValue::Rgb8U(c) => c,
        //                                     _ => unreachable!(),
        //                                 }
        //                             })
        //                         });

        //                     rle_with_cfg(data, rle_cfg)
        //                     // data
        //                 })
        //             });

        //             let (rle, color) = rle_data
        //                 .map(|rle| (rle.count, rle.value))
        //                 .unzip::<_, _, Vec<_>, Vec<_>>();

        //             let ctx_rle = jkl::ans::Context::from_input(rle);
        //             let ctx_color = jkl::ans::Context::from_input(color);

        //             let mut write_size = WriteSize::new();
        //             let mut write_bits = WriteBits::new(&mut write_size);
        //             ctx_rle.write(&mut write_bits).unwrap();
        //             write_bits.finish().unwrap();
        //             self.rans_size += write_size.size as usize * 8;

        //             let mut write_size = WriteSize::new();
        //             let mut write_bits = WriteBits::new(&mut write_size);
        //             ctx_color.write(&mut write_bits).unwrap();
        //             write_bits.finish().unwrap();
        //             self.rans_size += write_size.size as usize * 8;

        //             (0..image.width()).step_by(256).for_each(|x| {
        //                 (0..image.height()).step_by(256).for_each(|y| {
        //                     let data =
        //                         (x..usize::min(x + 256, image.width())).flat_map(move |dx| {
        //                             (y..usize::min(y + 256, image.height())).map(move |dy| {
        //                                 let p = image.get(dx, dy);

        //                                 match p {
        //                                     PixelValue::Rgb8U(c) => c,
        //                                     _ => unreachable!(),
        //                                 }
        //                             })
        //                         });

        //                     let rle_data = rle_with_cfg(data, rle_cfg);
        //                     // let rle_data = data;

        //                     let mut encoder_rle = jkl::ans::Encoder::new(&ctx_rle);
        //                     let mut encoder_color = jkl::ans::Encoder::new(&ctx_color);

        //                     rle_data.for_each(|p| {
        //                         if let Some(_) = encoder_rle.encode(p.count) {
        //                             self.rans_size += 32;
        //                         }
        //                         if let Some(_) = encoder_color.encode(p.value) {
        //                             self.rans_size += 32;
        //                         }
        //                     });

        //                     self.rans_size += 64;
        //                 })
        //             });
        //         }

        //         Some(PixelType::Rgba8U) => {
        //             let rle_data = (0..image.width()).step_by(256).flat_map(move |x| {
        //                 (0..image.height()).step_by(256).flat_map(move |y| {
        //                     let data =
        //                         (x..usize::min(x + 256, image.width())).flat_map(move |dx| {
        //                             (y..usize::min(y + 256, image.height())).map(move |dy| {
        //                                 let p = image.get(dx, dy);

        //                                 match p {
        //                                     PixelValue::Rgba8U(c) => c,
        //                                     _ => unreachable!(),
        //                                 }
        //                             })
        //                         });

        //                     rle_with_cfg(data, rle_cfg)
        //                     // data
        //                 })
        //             });

        //             let (rle, color) = rle_data
        //                 .map(|rle| (rle.count, rle.value))
        //                 .unzip::<_, _, Vec<_>, Vec<_>>();

        //             let ctx_rle = jkl::ans::Context::from_input(rle);
        //             let ctx_color = jkl::ans::Context::from_input(color);

        //             let mut write_size = WriteSize::new();
        //             let mut write_bits = WriteBits::new(&mut write_size);
        //             ctx_rle.write(&mut write_bits).unwrap();
        //             write_bits.finish().unwrap();
        //             self.rans_size += write_size.size as usize * 8;

        //             let mut write_size = WriteSize::new();
        //             let mut write_bits = WriteBits::new(&mut write_size);
        //             ctx_color.write(&mut write_bits).unwrap();
        //             write_bits.finish().unwrap();
        //             self.rans_size += write_size.size as usize * 8;

        //             (0..image.width()).step_by(256).for_each(|x| {
        //                 (0..image.height()).step_by(256).for_each(|y| {
        //                     let data =
        //                         (x..usize::min(x + 256, image.width())).flat_map(move |dx| {
        //                             (y..usize::min(y + 256, image.height())).map(move |dy| {
        //                                 let p = image.get(dx, dy);

        //                                 match p {
        //                                     PixelValue::Rgba8U(c) => c,
        //                                     _ => unreachable!(),
        //                                 }
        //                             })
        //                         });

        //                     let rle_data = rle_with_cfg(data, rle_cfg);
        //                     // let rle_data = data;

        //                     let mut encoder_rle = jkl::ans::Encoder::new(&ctx_rle);
        //                     let mut encoder_color = jkl::ans::Encoder::new(&ctx_color);

        //                     rle_data.for_each(|p| {
        //                         if let Some(_) = encoder_rle.encode(p.count) {
        //                             self.rans_size += 32;
        //                         }
        //                         if let Some(_) = encoder_color.encode(p.value) {
        //                             self.rans_size += 32;
        //                         }
        //                     });

        //                     self.rans_size += 64;
        //                 })
        //             });
        //         }

        //         Some(PixelType::BC1) => {
        //             let rle_data = (0..image.width()).step_by(256).flat_map(move |x| {
        //                 (0..image.height()).step_by(256).flat_map(move |y| {
        //                     let data =
        //                         (x..usize::min(x + 256, image.width())).flat_map(move |dx| {
        //                             (y..usize::min(y + 256, image.height())).map(move |dy| {
        //                                 let p = image.get(dx, dy);

        //                                 match p {
        //                                     PixelValue::BC1(b) => b,
        //                                     _ => unreachable!(),
        //                                 }
        //                             })
        //                         });

        //                     rle_with_cfg(data, rle_cfg)
        //                     // data
        //                 })
        //             });

        //             let (rle, color) = rle_data
        //                 .map(|rle| (rle.count, rle.value))
        //                 .unzip::<_, _, Vec<_>, Vec<_>>();

        //             let ctx_rle = jkl::ans::Context::from_input(rle);
        //             let ctx_color = jkl::ans::Context::from_input(color);

        //             let mut write_size = WriteSize::new();
        //             let mut write_bits = WriteBits::new(&mut write_size);
        //             ctx_rle.write(&mut write_bits).unwrap();
        //             write_bits.finish().unwrap();
        //             self.rans_size += write_size.size as usize * 8;

        //             let mut write_size = WriteSize::new();
        //             let mut write_bits = WriteBits::new(&mut write_size);
        //             ctx_color.write(&mut write_bits).unwrap();
        //             write_bits.finish().unwrap();
        //             self.rans_size += write_size.size as usize * 8;

        //             (0..image.width()).step_by(256).for_each(|x| {
        //                 (0..image.height()).step_by(256).for_each(|y| {
        //                     let data =
        //                         (x..usize::min(x + 256, image.width())).flat_map(move |dx| {
        //                             (y..usize::min(y + 256, image.height())).map(move |dy| {
        //                                 let p = image.get(dx, dy);

        //                                 match p {
        //                                     PixelValue::BC1(b) => b,
        //                                     _ => unreachable!(),
        //                                 }
        //                             })
        //                         });

        //                     let rle_data = rle_with_cfg(data, rle_cfg);
        //                     // let rle_data = data;

        //                     let mut encoder_rle = jkl::ans::Encoder::new(&ctx_rle);
        //                     let mut encoder_color = jkl::ans::Encoder::new(&ctx_color);

        //                     rle_data.for_each(|p| {
        //                         if let Some(_) = encoder_rle.encode(p.count) {
        //                             self.rans_size += 32;
        //                         }
        //                         if let Some(_) = encoder_color.encode(p.value) {
        //                             self.rans_size += 32;
        //                         }
        //                     });

        //                     self.rans_size += 64;
        //                 })
        //             });
        //         }
        //     }
        // } else {
        //     match self.input {
        //         None => {}
        //         Some(PixelType::Rgb8U) => {
        //             let rle_data = (0..image.width()).step_by(256).flat_map(move |x| {
        //                 (0..image.height()).step_by(256).flat_map(move |y| {
        //                     let data =
        //                         (x..usize::min(x + 256, image.width())).flat_map(move |dx| {
        //                             (y..usize::min(y + 256, image.height())).map(move |dy| {
        //                                 let p = image.get(dx, dy);

        //                                 match p {
        //                                     PixelValue::Rgb8U(c) => c,
        //                                     _ => unreachable!(),
        //                                 }
        //                             })
        //                         });

        //                     rle_with_cfg(data, rle_cfg)
        //                     // data
        //                 })
        //             });

        //             let ctx = jkl::ans::Context::from_input(rle_data);

        //             let mut write_size = WriteSize::new();
        //             let mut write_bits = WriteBits::new(&mut write_size);
        //             ctx.write(&mut write_bits).unwrap();
        //             write_bits.finish().unwrap();
        //             self.rans_size += write_size.size as usize * 8;

        //             (0..image.width()).step_by(256).for_each(|x| {
        //                 (0..image.height()).step_by(256).for_each(|y| {
        //                     let data =
        //                         (x..usize::min(x + 256, image.width())).flat_map(move |dx| {
        //                             (y..usize::min(y + 256, image.height())).map(move |dy| {
        //                                 let p = image.get(dx, dy);

        //                                 match p {
        //                                     PixelValue::Rgb8U(c) => c,
        //                                     _ => unreachable!(),
        //                                 }
        //                             })
        //                         });

        //                     let rle_data = rle_with_cfg(data, rle_cfg);
        //                     // let rle_data = data;

        //                     let mut encoder = jkl::ans::Encoder::new(&ctx);

        //                     rle_data.for_each(|p| {
        //                         if let Some(_) = encoder.encode(p) {
        //                             self.rans_size += 32;
        //                         }
        //                     });

        //                     self.rans_size += 64;
        //                 })
        //             });
        //         }

        //         Some(PixelType::Rgba8U) => {
        //             let rle_data = (0..image.width()).step_by(256).flat_map(move |x| {
        //                 (0..image.height()).step_by(256).flat_map(move |y| {
        //                     let data =
        //                         (x..usize::min(x + 256, image.width())).flat_map(move |dx| {
        //                             (y..usize::min(y + 256, image.height())).map(move |dy| {
        //                                 let p = image.get(dx, dy);

        //                                 match p {
        //                                     PixelValue::Rgba8U(c) => c,
        //                                     _ => unreachable!(),
        //                                 }
        //                             })
        //                         });

        //                     rle_with_cfg(data, rle_cfg)
        //                     // data
        //                 })
        //             });

        //             let ctx = jkl::ans::Context::from_input(rle_data);

        //             let mut write_size = WriteSize::new();
        //             let mut write_bits = WriteBits::new(&mut write_size);
        //             ctx.write(&mut write_bits).unwrap();
        //             write_bits.finish().unwrap();
        //             self.rans_size += write_size.size as usize * 8;

        //             (0..image.width()).step_by(256).for_each(|x| {
        //                 (0..image.height()).step_by(256).for_each(|y| {
        //                     let data =
        //                         (x..usize::min(x + 256, image.width())).flat_map(move |dx| {
        //                             (y..usize::min(y + 256, image.height())).map(move |dy| {
        //                                 let p = image.get(dx, dy);

        //                                 match p {
        //                                     PixelValue::Rgba8U(c) => c,
        //                                     _ => unreachable!(),
        //                                 }
        //                             })
        //                         });

        //                     let rle_data = rle_with_cfg(data, rle_cfg);
        //                     // let rle_data = data;

        //                     let mut encoder = jkl::ans::Encoder::new(&ctx);

        //                     rle_data.for_each(|p| {
        //                         if let Some(_) = encoder.encode(p) {
        //                             self.rans_size += 32;
        //                         }
        //                     });

        //                     self.rans_size += 64;
        //                 })
        //             });
        //         }

        //         Some(PixelType::BC1) => {
        //             let rle_data = (0..image.width()).step_by(256).flat_map(move |x| {
        //                 (0..image.height()).step_by(256).flat_map(move |y| {
        //                     let data =
        //                         (x..usize::min(x + 256, image.width())).flat_map(move |dx| {
        //                             (y..usize::min(y + 256, image.height())).map(move |dy| {
        //                                 let p = image.get(dx, dy);

        //                                 match p {
        //                                     PixelValue::BC1(b) => b,
        //                                     _ => unreachable!(),
        //                                 }
        //                             })
        //                         });

        //                     rle_with_cfg(data, rle_cfg)
        //                     // data
        //                 })
        //             });

        //             let ctx = jkl::ans::Context::from_input(rle_data);

        //             let mut write_size = WriteSize::new();
        //             let mut write_bits = WriteBits::new(&mut write_size);
        //             ctx.write(&mut write_bits).unwrap();
        //             write_bits.finish().unwrap();
        //             self.rans_size += write_size.size as usize * 8;

        //             (0..image.width()).step_by(256).for_each(|x| {
        //                 (0..image.height()).step_by(256).for_each(|y| {
        //                     let data =
        //                         (x..usize::min(x + 256, image.width())).flat_map(move |dx| {
        //                             (y..usize::min(y + 256, image.height())).map(move |dy| {
        //                                 let p = image.get(dx, dy);

        //                                 match p {
        //                                     PixelValue::BC1(b) => b,
        //                                     _ => unreachable!(),
        //                                 }
        //                             })
        //                         });

        //                     let rle_data = rle_with_cfg(data, rle_cfg);
        //                     // let rle_data = data;

        //                     let mut encoder = jkl::ans::Encoder::new(&ctx);

        //                     rle_data.for_each(|p| {
        //                         if let Some(_) = encoder.encode(p) {
        //                             self.rans_size += 32;
        //                         }
        //                     });

        //                     self.rans_size += 64;
        //                 })
        //             });
        //         }
        //     }
        // }
    }
}

impl serde::Serialize for RleRansCalculatorNode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_unit()
    }
}

impl<'de> serde::Deserialize<'de> for RleRansCalculatorNode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        <() as serde::Deserialize<'de>>::deserialize(deserializer)?;
        Ok(RleRansCalculatorNode::new())
    }
}

/// Builds reference map
struct ReferenceMapNode {
    size: (usize, usize),
    input: Option<Image<Rgb32F>>,
    map: ReferenceMap<Rgb32F>,
    body: ImageWidget,
    batch_size: usize,
    learning_rate: f32,
}

impl ReferenceMapNode {
    fn new() -> Self {
        ReferenceMapNode::with_size((0, 0))
    }

    fn with_size(size: (usize, usize)) -> Self {
        let mut map = ReferenceMap::new(size.0, size.1, Rgb32F::BLACK);
        map.random_initialize(|| {
            Rgb32F::new(
                rand::random_range(0.0..=1.0),
                rand::random_range(0.0..=1.0),
                rand::random_range(0.0..=1.0),
            )
        });

        ReferenceMapNode {
            size,
            input: None,
            map,
            body: ImageWidget::new(),
            batch_size: 16,
            learning_rate: 0.01,
        }
    }

    fn prepare(&mut self, ctx: &egui::Context) {
        self.body.make_texture(ctx, || egui::ColorImage {
            size: [self.size.0, self.size.1],
            source_size: egui::vec2(self.size.0 as f32, self.size.1 as f32),
            pixels: self
                .map
                .as_ref()
                .pixels()
                .iter()
                .map(|p| {
                    egui::Color32::from_rgb(
                        (p.r() * 255.0) as u8,
                        (p.g() * 255.0) as u8,
                        (p.b() * 255.0) as u8,
                    )
                })
                .collect(),
        })
    }

    fn title(&self) -> String {
        "Reference Map".to_owned()
    }

    fn inputs(&self) -> usize {
        3
    }

    fn set_input_ty(&mut self, input: usize, ty: JackalType) -> bool {
        match input {
            0 | 1 => match ty {
                JackalType::Uint => true,
                _ => false,
            },
            2 => match ty {
                JackalType::Null => {
                    self.input = None;
                    true
                }
                JackalType::Image(PixelType::Rgb8U) => {
                    self.input = None;
                    true
                }
                _ => false,
            },
            _ => unreachable!(),
        }
    }

    fn input_ty(&self, input: usize) -> JackalType {
        match input {
            0 | 1 => JackalType::Uint,
            2 => match &self.input {
                None => JackalType::Null,
                Some(_) => JackalType::Image(PixelType::Rgb8U),
            },
            _ => unreachable!(),
        }
    }

    fn input_ui(&mut self, input: usize, ui: &mut Ui) {
        let rebuild = match input {
            0 => ui
                .add(
                    egui::DragValue::new(&mut self.size.0)
                        .range(1..=8192)
                        .clamp_existing_to_range(true),
                )
                .changed(),
            1 => ui
                .add(
                    egui::DragValue::new(&mut self.size.1)
                        .range(1..=8192)
                        .clamp_existing_to_range(true),
                )
                .changed(),
            2 => false,
            _ => unreachable!(),
        };

        if rebuild {
            self.rebuild();
        }
    }

    fn set_input(&mut self, input: usize, value: JackalValue) {
        match input {
            0 => match value {
                JackalValue::Uint(value) => {
                    self.size.0 = value;
                }
                _ => unreachable!(),
            },
            1 => match value {
                JackalValue::Uint(value) => {
                    self.size.1 = value;
                }
                _ => unreachable!(),
            },
            2 => match value {
                JackalValue::Null => {
                    self.input = None;
                }
                JackalValue::Image(ImageValue::Rgb8U(image)) => {
                    self.input = Some(Image::new(
                        image.width,
                        image.height,
                        image.pixels.into_iter().map(Rgb8U::into_f32).collect(),
                    ));
                }
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }

        self.rebuild();
    }

    fn outputs(&self) -> usize {
        1
    }

    fn output_ty(&self, output: usize) -> JackalType {
        assert_eq!(output, 0);
        JackalType::Image(PixelType::Rgb8U)
    }

    fn output_ui(&mut self, output: usize, ui: &mut Ui) {
        assert_eq!(output, 0);
        ui.label("Rgb8U image");
    }

    fn get_output(&self, output: usize) -> JackalValue {
        assert_eq!(output, 0);
        JackalValue::Null
    }

    fn has_body(&self) -> bool {
        true
    }

    fn body_ui(&mut self, ui: &mut Ui) {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut self.batch_size)
                        .range(1..=1 << 20)
                        .clamp_existing_to_range(true),
                );
                ui.add(
                    egui::DragValue::new(&mut self.learning_rate)
                        .range(0.01..=10.0)
                        .speed(0.01)
                        .clamp_existing_to_range(true),
                );

                let r = ui.small_button("train");
                if r.clicked() {
                    self.train();
                }

                let r = ui.small_button("reset");
                if r.clicked() {
                    self.rebuild();
                }
            });

            self.body.show(ui);
        });
    }

    fn rebuild(&mut self) {
        self.body.unmake_texture();
        self.map = ReferenceMap::new(self.size.0, self.size.1, Rgb32F::BLACK);

        // match &self.input {
        //     None => {
        self.map.random_initialize(|| {
            Rgb32F::new(
                rand::random_range(0.0..=1.0),
                rand::random_range(0.0..=1.0),
                rand::random_range(0.0..=1.0),
            )
        });
        //     }
        //     Some(input) => {
        //         let block_size = 8;

        //         self.map
        //             .as_mut()
        //             .initialize_patches(input.as_ref(), block_size);
        //     }
        // };
    }

    fn train(&mut self) {
        let Some(input) = &self.input else {
            return;
        };

        self.body.unmake_texture();
        let block_size = 8;
        let batch_size = self.batch_size;

        self.map.train2(
            input.as_ref(),
            block_size,
            batch_size,
            |a, b| (Rgb32F::distance(a, b) + 1.0).log2(),
            |a, b, e| {
                let e = self.learning_rate / (1.0 + e).powi(1);
                Rgb32F::lerp(a, b, e)
            },
            // |a, b, e| {
            //     let e = self.learning_rate * e / block_size as f32 / block_size as f32;
            //     Rgb32F::lerp(a, b, e)
            // },
        );
    }
}

impl serde::Serialize for ReferenceMapNode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        (self.size.0 as usize, self.size.1 as usize).serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for ReferenceMapNode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let size = <(usize, usize) as serde::Deserialize<'de>>::deserialize(deserializer)?;
        Ok(ReferenceMapNode::with_size((size.0, size.1)))
    }
}

/// Finds simlar patches in earlier encoded image part
/// writes reference to the patch and residual.
struct BlockCopyNode {
    input: Option<Image<Rgb8U>>,
    output: Option<Image<Rgb8U>>,
    ref_size: usize,
    block_size: usize,
    window_size: usize,
    body: ImageWidget,
}

impl BlockCopyNode {
    fn new() -> Self {
        BlockCopyNode {
            input: None,
            output: None,
            ref_size: 0,
            block_size: 8,
            window_size: 8,
            body: ImageWidget::new(),
        }
    }

    fn prepare(&mut self, ctx: &egui::Context) {
        let Some(output) = &self.output else {
            return;
        };

        self.body.make_texture(ctx, || output.to_egui())
    }

    fn title(&self) -> String {
        "Block Copy".to_owned()
    }

    fn inputs(&self) -> usize {
        3
    }

    fn set_input_ty(&mut self, input: usize, ty: JackalType) -> bool {
        match input {
            0 | 1 => match ty {
                JackalType::Uint => true,
                _ => false,
            },
            2 => match ty {
                JackalType::Null => {
                    self.input = None;
                    true
                }
                JackalType::Image(PixelType::Rgb8U) => {
                    self.input = None;
                    true
                }
                _ => false,
            },
            _ => unreachable!(),
        }
    }

    fn input_ty(&self, input: usize) -> JackalType {
        match input {
            0 | 1 => JackalType::Uint,
            2 => match &self.input {
                None => JackalType::Null,
                Some(_) => JackalType::Image(PixelType::Rgb8U),
            },
            _ => unreachable!(),
        }
    }

    fn input_ui(&mut self, input: usize, ui: &mut Ui) {
        let rebuild = match input {
            0 => ui
                .add(
                    egui::DragValue::new(&mut self.block_size)
                        .range(1..=8192)
                        .clamp_existing_to_range(true),
                )
                .changed(),
            1 => ui
                .add(
                    egui::DragValue::new(&mut self.window_size)
                        .range(1..=8192)
                        .clamp_existing_to_range(true),
                )
                .changed(),
            2 => false,
            _ => unreachable!(),
        };

        if rebuild {
            self.rebuild();
        }
    }

    fn set_input(&mut self, input: usize, value: JackalValue) {
        match input {
            0 => match value {
                JackalValue::Uint(value) => {
                    self.block_size = value;
                }
                _ => unreachable!(),
            },
            1 => match value {
                JackalValue::Uint(value) => {
                    self.window_size = value;
                }
                _ => unreachable!(),
            },
            2 => match value {
                JackalValue::Null => {
                    self.input = None;
                }
                JackalValue::Image(ImageValue::Rgb8U(image)) => {
                    self.input = Some(image);
                }
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }

        self.rebuild();
    }

    fn outputs(&self) -> usize {
        2
    }

    fn output_ty(&self, output: usize) -> JackalType {
        match output {
            0 => JackalType::Uint,
            1 => JackalType::Image(PixelType::Rgb8U),
            _ => unreachable!(),
        }
    }

    fn output_ui(&mut self, output: usize, ui: &mut Ui) {
        match output {
            0 => {
                let human_readable = human_readable_bits(self.ref_size);
                let r = ui.label(human_readable);
                r.on_hover_ui(|ui| {
                    ui.label(format!("{} bit", self.ref_size));
                });
            }
            1 => {
                ui.label("Rgb8U image");
            }
            _ => unreachable!(),
        }
    }

    fn get_output(&self, output: usize) -> JackalValue {
        match output {
            0 => JackalValue::Uint(self.ref_size as usize),
            1 => match &self.output {
                None => JackalValue::Null,
                Some(image) => JackalValue::Image(ImageValue::Rgb8U(image.clone())),
            },
            _ => unreachable!(),
        }
    }

    fn has_body(&self) -> bool {
        true
    }

    fn body_ui(&mut self, ui: &mut Ui) {
        self.body.show(ui);
    }

    fn rebuild(&mut self) {
        self.body.unmake_texture();

        let Some(input) = &self.input else {
            return;
        };

        let mut output = match &mut self.output {
            Some(output) if output.width == input.width && output.height == input.height => {
                output.as_mut()
            }
            _ => {
                self.output = None;
                let image = Image::solid(input.width, input.height, Rgb8U::BLACK);
                self.output.get_or_insert(image).as_mut()
            }
        };

        let mut write_size = WriteSize::new();
        let mut write_bits = WriteBits::new(&mut write_size);

        for y in (0..input.height).step_by(self.block_size) {
            for x in (0..input.width).step_by(self.block_size) {
                if x == 0 && y == 0 {
                    // skip first block
                    continue;
                }

                let bw = usize::min(self.block_size, input.width - x);
                let bh = usize::min(self.block_size, input.height - y);

                let from_x = x.saturating_sub(self.window_size);
                let to_x = (x + bw + self.window_size).min(input.width);
                let from_y = y.saturating_sub(self.window_size);

                let block = input.as_ref().get_range(x, y, bw, bh);

                let (best_error1, best_match1) = if y > 0 {
                    let to_y = y + bh - 1;

                    let (error, (mx, my)) = input
                        .as_ref()
                        .get_range(from_x, from_y, to_x - from_x, to_y - from_y)
                        .find_best_match(1, 1, block, Rgb8U::distance);

                    (error, (mx + from_x, my + from_y))
                } else {
                    (f32::INFINITY, (0, 0))
                };

                let (best_error2, best_match2) = if x > 0 {
                    let to_x = x + bw - 1;
                    let from_y = y;
                    let to_y = y + bh;

                    let (error, (mx, my)) = input
                        .as_ref()
                        .get_range(from_x, from_y, to_x - from_x, to_y - from_y)
                        .find_best_match(1, 1, block, Rgb8U::distance);

                    (error, (mx + from_x, my + from_y))
                } else {
                    (f32::INFINITY, (0, 0))
                };

                assert!(best_error1.is_finite() || best_error2.is_finite());

                let out_block = output.get_range_mut(x, y, bw, bh);

                if best_error1 < best_error2 {
                    input.as_ref().residual(
                        best_match1.0,
                        best_match1.1,
                        block,
                        out_block,
                        Rgb8U::wrapping_sub,
                    );

                    vle::encode(
                        (x as isize - best_match1.0 as isize).zigzag(),
                        &mut write_bits,
                    )
                    .unwrap();
                    vle::encode(y - best_match1.1, &mut write_bits).unwrap();
                } else {
                    input.as_ref().residual(
                        best_match2.0,
                        best_match2.1,
                        block,
                        out_block,
                        Rgb8U::wrapping_sub,
                    );

                    vle::encode(
                        (x as isize - best_match2.0 as isize).zigzag(),
                        &mut write_bits,
                    )
                    .unwrap();
                    vle::encode(y - best_match2.1, &mut write_bits).unwrap();
                }
            }
        }

        self.ref_size = write_size.size as usize * 8;
    }
}

impl serde::Serialize for BlockCopyNode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        <() as serde::Serialize>::serialize(&(), serializer)
    }
}

impl<'de> serde::Deserialize<'de> for BlockCopyNode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        <() as serde::Deserialize<'de>>::deserialize(deserializer)?;
        Ok(BlockCopyNode::new())
    }
}

fn rgb8u_ord(a: Rgb8U, b: Rgb8U) -> std::cmp::Ordering {
    a.bits_interleaved().cmp(&b.bits_interleaved())
}

fn rgb8u_delta(a: Rgb8U, b: Rgb8U) -> Vle<u32> {
    Vle(b.bits_interleaved() - a.bits_interleaved())
}

fn rgba8u_ord(a: Rgba8U, b: Rgba8U) -> std::cmp::Ordering {
    a.bits_interleaved().cmp(&b.bits_interleaved())
}

fn rgba8u_delta(a: Rgba8U, b: Rgba8U) -> Vle<u32> {
    Vle(b.bits_interleaved() - a.bits_interleaved())
}

fn rgb565_ord(a: Rgb565, b: Rgb565) -> std::cmp::Ordering {
    a.bits_interleaved().cmp(&b.bits_interleaved())
}

fn rgb565_delta(a: Rgb565, b: Rgb565) -> Vle<u16> {
    Vle(b.bits_interleaved() - a.bits_interleaved())
}

fn bc1_ord(a: bc1::Block, b: bc1::Block) -> std::cmp::Ordering {
    let a0 = a.color0.bits_interleaved();
    let a1 = a.color1.bits_interleaved();
    let a = interleave16_2(a0, a1);
    let b0 = b.color0.bits_interleaved();
    let b1 = b.color1.bits_interleaved();
    let b = interleave16_2(b0, b1);
    a.cmp(&b)
}

fn bc1_delta(a: bc1::Block, b: bc1::Block) -> Vle<u64> {
    let a0 = a.color0.bits_interleaved();
    let a1 = a.color1.bits_interleaved();
    let ai = interleave16_2(a0, a1);
    let b0 = b.color0.bits_interleaved();
    let b1 = b.color1.bits_interleaved();
    let bi = interleave16_2(b0, b1);

    let hi = bi - ai;
    let lo = u32::from_le_bytes(b.texels);
    Vle(((hi as u64) << 32) | lo as u64)
}

fn lzrgb8u_ord(a: lz77::Token<Rgb8U>, b: lz77::Token<Rgb8U>) -> std::cmp::Ordering {
    match (a, b) {
        (lz77::Token::Literal { symbol: a }, lz77::Token::Literal { symbol: b }) => rgb8u_ord(a, b),
        (lz77::Token::Literal { .. }, lz77::Token::Reference { .. }) => std::cmp::Ordering::Less,
        (lz77::Token::Reference { .. }, lz77::Token::Literal { .. }) => std::cmp::Ordering::Greater,
        (
            lz77::Token::Reference {
                length: al,
                distance: ad,
            },
            lz77::Token::Reference {
                length: bl,
                distance: bd,
            },
        ) => (al, ad).cmp(&(bl, bd)),
    }
}

fn lzrgb8u_delta(a: lz77::Token<Rgb8U>, b: lz77::Token<Rgb8U>) -> lz77::Token<Vle<u32>> {
    match (a, b) {
        (lz77::Token::Literal { symbol: a }, lz77::Token::Literal { symbol: b }) => {
            lz77::Token::Literal {
                symbol: rgb8u_delta(a, b),
            }
        }
        (lz77::Token::Literal { .. }, lz77::Token::Reference { length, distance }) => {
            lz77::Token::Reference { length, distance }
        }
        (lz77::Token::Reference { .. }, lz77::Token::Literal { .. }) => {
            unreachable!()
        }
        (
            lz77::Token::Reference {
                length: al,
                distance: ad,
            },
            lz77::Token::Reference {
                length: bl,
                distance: bd,
            },
        ) => {
            assert!(bl >= al);

            lz77::Token::Reference {
                length: bl - al + 2,
                distance: if bl > al { bd } else { bd - ad },
            }
        }
    }
}

fn lzrgba8u_ord(a: lz77::Token<Rgba8U>, b: lz77::Token<Rgba8U>) -> std::cmp::Ordering {
    match (a, b) {
        (lz77::Token::Literal { symbol: a }, lz77::Token::Literal { symbol: b }) => {
            rgba8u_ord(a, b)
        }
        (lz77::Token::Literal { .. }, lz77::Token::Reference { .. }) => std::cmp::Ordering::Less,
        (lz77::Token::Reference { .. }, lz77::Token::Literal { .. }) => std::cmp::Ordering::Greater,
        (
            lz77::Token::Reference {
                length: al,
                distance: ad,
            },
            lz77::Token::Reference {
                length: bl,
                distance: bd,
            },
        ) => (al, ad).cmp(&(bl, bd)),
    }
}

fn lzrgba8u_delta(a: lz77::Token<Rgba8U>, b: lz77::Token<Rgba8U>) -> lz77::Token<Vle<u32>> {
    match (a, b) {
        (lz77::Token::Literal { symbol: a }, lz77::Token::Literal { symbol: b }) => {
            lz77::Token::Literal {
                symbol: rgba8u_delta(a, b),
            }
        }
        (lz77::Token::Literal { .. }, lz77::Token::Reference { length, distance }) => {
            lz77::Token::Reference { length, distance }
        }
        (lz77::Token::Reference { .. }, lz77::Token::Literal { .. }) => {
            unreachable!()
        }
        (
            lz77::Token::Reference {
                length: al,
                distance: ad,
            },
            lz77::Token::Reference {
                length: bl,
                distance: bd,
            },
        ) => {
            assert!(bl >= al);

            lz77::Token::Reference {
                length: bl - al + 2,
                distance: if bl > al { bd } else { bd - ad },
            }
        }
    }
}

fn lzrgb565_ord(a: lz77::Token<Rgb565>, b: lz77::Token<Rgb565>) -> std::cmp::Ordering {
    match (a, b) {
        (lz77::Token::Literal { symbol: a }, lz77::Token::Literal { symbol: b }) => {
            rgb565_ord(a, b)
        }
        (lz77::Token::Literal { .. }, lz77::Token::Reference { .. }) => std::cmp::Ordering::Less,
        (lz77::Token::Reference { .. }, lz77::Token::Literal { .. }) => std::cmp::Ordering::Greater,
        (
            lz77::Token::Reference {
                length: al,
                distance: ad,
            },
            lz77::Token::Reference {
                length: bl,
                distance: bd,
            },
        ) => (al, ad).cmp(&(bl, bd)),
    }
}

fn lzrgb565_delta(a: lz77::Token<Rgb565>, b: lz77::Token<Rgb565>) -> lz77::Token<Vle<u16>> {
    match (a, b) {
        (lz77::Token::Literal { symbol: a }, lz77::Token::Literal { symbol: b }) => {
            lz77::Token::Literal {
                symbol: rgb565_delta(a, b),
            }
        }
        (lz77::Token::Literal { .. }, lz77::Token::Reference { length, distance }) => {
            lz77::Token::Reference { length, distance }
        }
        (lz77::Token::Reference { .. }, lz77::Token::Literal { .. }) => {
            unreachable!()
        }
        (
            lz77::Token::Reference {
                length: al,
                distance: ad,
            },
            lz77::Token::Reference {
                length: bl,
                distance: bd,
            },
        ) => {
            assert!(bl >= al);

            lz77::Token::Reference {
                length: bl - al + 2,
                distance: if bl > al { bd } else { bd - ad },
            }
        }
    }
}

fn lzbc1_ord(a: lz77::Token<bc1::Block>, b: lz77::Token<bc1::Block>) -> std::cmp::Ordering {
    match (a, b) {
        (lz77::Token::Literal { symbol: a }, lz77::Token::Literal { symbol: b }) => bc1_ord(a, b),
        (lz77::Token::Literal { .. }, lz77::Token::Reference { .. }) => std::cmp::Ordering::Less,
        (lz77::Token::Reference { .. }, lz77::Token::Literal { .. }) => std::cmp::Ordering::Greater,
        (
            lz77::Token::Reference {
                length: al,
                distance: ad,
            },
            lz77::Token::Reference {
                length: bl,
                distance: bd,
            },
        ) => (al, ad).cmp(&(bl, bd)),
    }
}

fn lzbc1_delta(a: lz77::Token<bc1::Block>, b: lz77::Token<bc1::Block>) -> lz77::Token<Vle<u64>> {
    match (a, b) {
        (lz77::Token::Literal { symbol: a }, lz77::Token::Literal { symbol: b }) => {
            lz77::Token::Literal {
                symbol: bc1_delta(a, b),
            }
        }
        (lz77::Token::Literal { .. }, lz77::Token::Reference { length, distance }) => {
            lz77::Token::Reference { length, distance }
        }
        (lz77::Token::Reference { .. }, lz77::Token::Literal { .. }) => {
            unreachable!()
        }
        (
            lz77::Token::Reference {
                length: al,
                distance: ad,
            },
            lz77::Token::Reference {
                length: bl,
                distance: bd,
            },
        ) => {
            assert!(bl >= al);

            lz77::Token::Reference {
                length: bl - al + 2,
                distance: if bl > al { bd } else { bd - ad },
            }
        }
    }
}
