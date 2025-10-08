use std::{marker::PhantomData, path::PathBuf};

use egui::{
    emath::TSTransform, load::SizedTexture, output, CentralPanel, Color32, ColorImage, Pos2,
    Stroke, TextureHandle, TextureOptions, Ui, Vec2,
};
use egui_snarl::{
    ui::{self, PinInfo, SnarlViewer, SnarlWidget},
    InPin, OutPin, Snarl,
};
use jkl::math::{Rgb32F, Rgb8U, Rgba8U, Vec3, Vec4};
use serde::{de, Deserialize};

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
            let r = ui.button("Add Paeth Node");
            let r = r.on_hover_text("Add a Paeth image filter node");
            if r.clicked() {
                snarl.insert_node(pos, JackalNode::Filter(FilterNode::new(Filter::Paeth)));
            }

            let r = ui.button("Add Strip Alpha Node");
            let r = r.on_hover_text("Add an Strip Alpha filter node");
            if r.clicked() {
                snarl.insert_node(pos, JackalNode::Filter(FilterNode::new(Filter::StripAlpha)));
            }

            let r = ui.button("Add LZP calculator Node");
            let r = r.on_hover_text("Add an LZP calculator node");
            if r.clicked() {
                snarl.insert_node(pos, JackalNode::LZPCalculator(LZPCalculator::new()));
            }
        });
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PixelType {
    Rgb8U,
    Rgba8U,
}

impl PixelType {
    fn name(&self) -> &'static str {
        match *self {
            PixelType::Rgb8U => "Rgb8U",
            PixelType::Rgba8U => "Rgba8U",
        }
    }

    fn default(&self) -> PixelValue {
        match self {
            PixelType::Rgb8U => PixelValue::Rgb8U(Rgb8U::BLACK),
            PixelType::Rgba8U => PixelValue::Rgba8U(Rgba8U::BLACK),
        }
    }

    fn bit_size(&self) -> u64 {
        match self {
            PixelType::Rgb8U => 24,
            PixelType::Rgba8U => 32,
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
            JackalType::Pixel(PixelType::Rgba8U) => Color32::LIGHT_BLUE,
            JackalType::Image(PixelType::Rgb8U) => Color32::GREEN,
            JackalType::Image(PixelType::Rgba8U) => Color32::LIGHT_GREEN,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum PixelValue {
    Rgb8U(Rgb8U),
    Rgba8U(Rgba8U),
}

impl PixelValue {
    fn pixel_ty(&self) -> PixelType {
        match *self {
            PixelValue::Rgb8U(_) => PixelType::Rgb8U,
            PixelValue::Rgba8U(_) => PixelType::Rgba8U,
        }
    }

    fn ty(&self) -> JackalType {
        JackalType::Pixel(self.pixel_ty())
    }

    fn hash(&self) -> u64 {
        match self {
            PixelValue::Rgb8U(pixel) => {
                (pixel.r() as u64) * 43 + (pixel.g() as u64) * 31 + (pixel.b() as u64) * 29
            }
            PixelValue::Rgba8U(pixel) => {
                (pixel.r() as u64) * 43
                    + (pixel.g() as u64) * 31
                    + (pixel.b() as u64) * 29
                    + (pixel.a() as u64) * 83
            }
        }
    }
}

#[derive(Clone)]
enum ImageValue {
    Rgb8U(Image<Rgb8U>),
    Rgba8U(Image<Rgba8U>),
}

impl ImageValue {
    pub fn new(width: u32, height: u32, pixel_type: PixelType) -> Self {
        match pixel_type {
            PixelType::Rgb8U => ImageValue::Rgb8U(Image::new(width, height, Rgb8U::BLACK)),
            PixelType::Rgba8U => ImageValue::Rgba8U(Image::new(width, height, Rgba8U::BLACK)),
        }
    }

    fn pixel_ty(&self) -> PixelType {
        match *self {
            ImageValue::Rgb8U(_) => PixelType::Rgb8U,
            ImageValue::Rgba8U(_) => PixelType::Rgba8U,
        }
    }

    fn ty(&self) -> JackalType {
        JackalType::Image(self.pixel_ty())
    }

    fn pixel_name(&self) -> &'static str {
        self.pixel_ty().name()
    }

    fn width(&self) -> u32 {
        match self {
            ImageValue::Rgb8U(image) => image.width,
            ImageValue::Rgba8U(image) => image.width,
        }
    }

    fn height(&self) -> u32 {
        match self {
            ImageValue::Rgb8U(image) => image.height,
            ImageValue::Rgba8U(image) => image.height,
        }
    }

    fn to_egui(&self) -> egui::ColorImage {
        match self {
            ImageValue::Rgb8U(image) => image.to_egui(),
            ImageValue::Rgba8U(image) => image.to_egui(),
        }
    }

    fn get(&self, x: u32, y: u32) -> PixelValue {
        match self {
            ImageValue::Rgb8U(image) => PixelValue::Rgb8U(image.get(x, y)),
            ImageValue::Rgba8U(image) => PixelValue::Rgba8U(image.get(x, y)),
        }
    }

    fn set(&mut self, x: u32, y: u32, pixel: PixelValue) {
        match (self, pixel) {
            (ImageValue::Rgb8U(image), PixelValue::Rgb8U(pixel)) => image.set(x, y, pixel),
            (ImageValue::Rgba8U(image), PixelValue::Rgba8U(pixel)) => image.set(x, y, pixel),
            (_, _) => panic!("Wrong pixel type"),
        }
    }
}

#[derive(Clone)]
enum JackalValue {
    Null,
    Uint(u64),
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
    LZPCalculator(LZPCalculator),
}

impl JackalNode {
    fn title(&self) -> String {
        match self {
            JackalNode::Dummy => "Dummy Node".into(),
            JackalNode::SourceImage(node) => node.title(),
            JackalNode::Filter(node) => node.title(),
            JackalNode::LZPCalculator(node) => node.title(),
        }
    }

    fn inputs(&self) -> usize {
        match self {
            JackalNode::Dummy => 1,
            JackalNode::SourceImage(node) => node.inputs(),
            JackalNode::Filter(node) => node.inputs(),
            JackalNode::LZPCalculator(node) => node.inputs(),
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
        }
    }

    fn outputs(&self) -> usize {
        match self {
            JackalNode::Dummy => 1,
            JackalNode::SourceImage(node) => node.outputs(),
            JackalNode::Filter(node) => node.outputs(),
            JackalNode::LZPCalculator(node) => node.outputs(),
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
        }
    }

    fn has_body(&self) -> bool {
        match self {
            JackalNode::Dummy => false,
            JackalNode::SourceImage(node) => node.has_body(),
            JackalNode::Filter(node) => node.has_body(),
            JackalNode::LZPCalculator(node) => node.has_body(),
        }
    }

    fn body_ui(&mut self, ui: &mut Ui) {
        match self {
            JackalNode::Dummy => unreachable!("Dummy node has no body"),
            JackalNode::SourceImage(node) => node.body_ui(ui),
            JackalNode::Filter(node) => node.body_ui(ui),
            JackalNode::LZPCalculator(node) => node.body_ui(ui),
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
        }
    }

    fn prepare(&mut self, ctx: &egui::Context) {
        match self {
            JackalNode::Dummy => {}
            JackalNode::SourceImage(node) => node.prepare(ctx),
            JackalNode::Filter(node) => node.prepare(ctx),
            JackalNode::LZPCalculator(node) => node.prepare(ctx),
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
        format!("Image: '{}'", self.file.to_string_lossy())
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

        let input = self.input.unwrap();
        let output = self.filter.convert_type(input);

        let output =
            self.output
                .get_or_insert(ImageValue::new(image.width(), image.height(), output));

        for x in 0..image.width() {
            for y in 0..image.height() {
                let a = if x == 0 {
                    input.default()
                } else {
                    image.get(x - 1, y)
                };

                let b = if y == 0 {
                    input.default()
                } else {
                    image.get(x, y - 1)
                };

                let c = if x == 0 || y == 0 {
                    input.default()
                } else {
                    image.get(x - 1, y - 1)
                };

                let t = image.get(x, y);
                let r = self.filter.filter(a, b, c, t);

                output.set(x, y, r);
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
}

impl Filter {
    fn name(&self) -> &'static str {
        match self {
            Filter::StripAlpha => "Strip Alpha",
            Filter::Paeth => "Paeth",
        }
    }

    fn convert_type(&self, input: PixelType) -> PixelType {
        match self {
            Filter::StripAlpha => match input {
                PixelType::Rgb8U => PixelType::Rgb8U,
                PixelType::Rgba8U => PixelType::Rgb8U,
            },
            Filter::Paeth => match input {
                PixelType::Rgb8U => PixelType::Rgb8U,
                PixelType::Rgba8U => PixelType::Rgba8U,
            },
        }
    }

    fn filter(&self, a: PixelValue, b: PixelValue, c: PixelValue, t: PixelValue) -> PixelValue {
        match self {
            Filter::StripAlpha => match t {
                PixelValue::Rgb8U(t) => PixelValue::Rgb8U(t),
                PixelValue::Rgba8U(t) => PixelValue::Rgb8U(t.rgb()),
            },
            Filter::Paeth => match (a, b, c, t) {
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
                _ => unreachable!(),
            },
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

    let r = Rgb8U::wrapping_sub(t, p);
    Rgb8U::wrapping_add(r, Rgb8U::new(128, 128, 128))
}

fn paeth_rgba(a: Rgba8U, b: Rgba8U, c: Rgba8U, t: Rgba8U) -> Rgba8U {
    let af = Vec4::new(a.r() as f32, a.g() as f32, a.b() as f32, a.a() as f32);
    let bf = Vec4::new(b.r() as f32, b.g() as f32, b.b() as f32, b.a() as f32);
    let cf = Vec4::new(c.r() as f32, c.g() as f32, c.b() as f32, c.a() as f32);

    let pf = af + bf - cf;

    let ad = Vec4::dot(pf - af, pf - af);
    let bd = Vec4::dot(pf - bf, pf - bf);
    let cd = Vec4::dot(pf - cf, pf - cf);

    let p = if ad <= bd && ad <= cd {
        a
    } else if bd <= cd {
        b
    } else {
        c
    };

    let r = Rgba8U::wrapping_sub(t, p);
    Rgba8U::wrapping_add(r, Rgba8U::new(128, 128, 128, 128))
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

fn convert_image(image: image::DynamicImage) -> ImageValue {
    match image {
        image::DynamicImage::ImageRgb8(rgb_image) => ImageValue::Rgb8U(Image {
            width: rgb_image.width(),
            height: rgb_image.height(),
            pixels: rgb_image.pixels().map(|p| rgb_image_to_jkl(*p)).collect(),
        }),
        image::DynamicImage::ImageRgba8(rgba_image) => ImageValue::Rgba8U(Image {
            width: rgba_image.width(),
            height: rgba_image.height(),
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
    width: u32,
    height: u32,
    pixels: Vec<T>,
}

impl<T> Image<T>
where
    T: Copy,
{
    fn new(width: u32, height: u32, fill: T) -> Self {
        Image {
            width,
            height,
            pixels: vec![fill; (width * height) as usize],
        }
    }

    fn get(&self, x: u32, y: u32) -> T {
        self.pixels[(y * self.width + x) as usize]
    }

    fn set(&mut self, x: u32, y: u32, value: T) {
        self.pixels[(y * self.width + x) as usize] = value;
    }
}

impl Image<Rgb8U> {
    fn to_egui(&self) -> egui::ColorImage {
        egui::ColorImage {
            size: [self.width as usize, self.height as usize],
            source_size: egui::Vec2::new(self.width as f32, self.height as f32),
            pixels: self.pixels.iter().copied().map(rgb8u_to_egui).collect(),
        }
    }
}

impl Image<Rgba8U> {
    fn to_egui(&self) -> egui::ColorImage {
        egui::ColorImage {
            size: [self.width as usize, self.height as usize],
            source_size: egui::Vec2::new(self.width as f32, self.height as f32),
            pixels: self.pixels.iter().copied().map(rgba8u_to_egui).collect(),
        }
    }
}

struct ImageWidget {
    texture: Option<TextureHandle>,
    max_size: Vec2,
}

impl ImageWidget {
    fn new() -> Self {
        Self {
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
            let size = texture.size_vec2().clamp(Vec2::splat(0.0), self.max_size);
            ui.image(SizedTexture::new(texture, size));
        }
    }
}

/// Calculates output size of LZP compression.
struct LZPCalculator {
    input: Option<PixelType>,
    lzp_size: u64,
}

impl LZPCalculator {
    fn new() -> Self {
        LZPCalculator {
            input: None,
            lzp_size: 0,
        }
    }

    fn prepare(&mut self, _ctx: &egui::Context) {}

    fn title(&self) -> String {
        "LZP calculator".to_owned()
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

        self.lzp_size = 0;

        let image = match value {
            JackalValue::Null => return,
            JackalValue::Image(image) if Some(image.pixel_ty()) == self.input => image,
            _ => unreachable!(),
        };

        let input = self.input.unwrap();

        let mut window = [input.default(); 3];
        let mut cache = [input.default(); 0x100];

        let mut total_bits = 0;

        for x in 0..image.width() {
            for y in 0..image.height() {
                let t = image.get(x, y);

                let h = window[0].hash() * 7 + window[1].hash() * 13 + window[2].hash() * 17;
                let p = cache[(h & 0xFF) as usize];
                cache[(h & 0xFF) as usize] = t;

                if p == t {
                    total_bits += 1;
                } else {
                    total_bits += 1;
                    total_bits += input.bit_size();
                }

                window.rotate_left(1);
                window[2] = t;
            }
        }

        self.lzp_size = total_bits;
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
        match self.input {
            Some(_) => {
                ui.label(format!("LZP compressed output is {} bits", self.lzp_size));
            }
            None => {
                ui.colored_label(Color32::RED, "No image");
            }
        }
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

    fn body_ui(&mut self, ui: &mut Ui) {
        unreachable!()
    }
}

impl serde::Serialize for LZPCalculator {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_unit()
    }
}

impl<'de> serde::Deserialize<'de> for LZPCalculator {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        <() as Deserialize<'de>>::deserialize(deserializer)?;
        Ok(LZPCalculator::new())
    }
}
