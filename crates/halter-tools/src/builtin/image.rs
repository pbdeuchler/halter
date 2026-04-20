// pattern: Imperative Shell

use std::io::Cursor;
use std::path::Path;

use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolResult, ToolSpec};
use image::codecs::jpeg::JpegEncoder;
use image::codecs::webp::WebPEncoder;
use image::imageops::FilterType;
use image::{DynamicImage, ImageFormat as ImageCrateFormat, io::Reader as ImageReader};
use serde_json::{Value, json};

use crate::{Tool, ToolContext};

use super::common::{
    ToolScope, atomic_write_blocking, ensure_not_cancelled, optional_string, optional_u64,
    required_string, resolve_path,
};

#[derive(Debug)]
pub struct ImageTool;

#[async_trait]
impl Tool for ImageTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("image"),
            description: "Inspect, resize, or convert raster image files".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["info", "resize", "convert"] },
                    "path": { "type": "string" },
                    "output_path": { "type": "string" },
                    "width": { "type": "integer", "minimum": 1 },
                    "height": { "type": "integer", "minimum": 1 },
                    "format": { "type": "string", "enum": ["png", "jpeg", "jpg", "webp", "gif"] },
                    "quality": { "type": "integer", "minimum": 1, "maximum": 100 },
                    "filter": {
                        "type": "string",
                        "enum": ["nearest", "triangle", "catmull-rom", "gaussian", "lanczos3"]
                    }
                },
                "required": ["action", "path"],
            }),
            concurrency: ToolConcurrency::Exclusive,
            capabilities: ToolCapabilities {
                mutating: true,
                requires_approval: false,
                cancellable: false,
                long_running: true,
            },
            provider_aliases: Default::default(),
        }
    }

    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        let _scope = ToolScope::new(&context, "image");
        ensure_not_cancelled(&context.cancel)?;
        let action = required_string(&input, "action")?.to_owned();
        let input_path = resolve_path(&context.working_dir, required_string(&input, "path")?);
        let output_path = optional_string(&input, "output_path")
            .map(|path| resolve_path(&context.working_dir, path));
        let width = optional_u64(&input, "width")?
            .map(u32::try_from)
            .transpose()
            .map_err(|_| anyhow::anyhow!("failed to execute image tool: width is out of range"))?;
        let height = optional_u64(&input, "height")?
            .map(u32::try_from)
            .transpose()
            .map_err(|_| anyhow::anyhow!("failed to execute image tool: height is out of range"))?;
        let quality = optional_u64(&input, "quality")?.unwrap_or(80).clamp(1, 100);
        let quality = u8::try_from(quality).map_err(|_| {
            anyhow::anyhow!("failed to execute image tool: quality is out of range")
        })?;
        let filter = parse_filter(optional_string(&input, "filter"))?;
        let requested_format = optional_string(&input, "format")
            .map(parse_format)
            .transpose()?
            .or_else(|| {
                output_path
                    .as_ref()
                    .and_then(|path| infer_format_from_path(path))
            });

        let input_len = std::fs::metadata(&input_path)
            .ok()
            .and_then(|metadata| usize::try_from(metadata.len()).ok())
            .unwrap_or(usize::MAX);
        let canonical_input = context
            .policy
            .check_read_path(&input_path, input_len)
            .await?;
        let input_path = canonical_input.into_path();
        let output_path = if let Some(output_path) = output_path {
            let canonical_output = context.policy.check_write_path(&output_path).await?;
            Some(canonical_output.into_path())
        } else {
            None
        };

        let path_locks = context.path_locks.clone();
        let result = tokio::task::spawn_blocking(move || {
            let bytes = if output_path
                .as_ref()
                .is_some_and(|output_path| output_path == &input_path)
            {
                let _lock = path_locks.acquire_write(&input_path)?;
                std::fs::read(&input_path)?
            } else {
                let _lock = path_locks.acquire_read(&input_path)?;
                std::fs::read(&input_path)?
            };
            let detected_format = image::guess_format(&bytes).ok().map(ImageFormat::from);
            let image = decode_image(&bytes)?;
            match action.as_str() {
                "info" => Ok(json!({
                    "path": input_path,
                    "format": requested_format
                        .or(detected_format)
                        .map(ImageFormat::as_str),
                    "width": image.width(),
                    "height": image.height(),
                })),
                "resize" => {
                    let width = width.ok_or_else(|| {
                        anyhow::anyhow!("failed to execute image tool: resize requires width")
                    })?;
                    let height = height.ok_or_else(|| {
                        anyhow::anyhow!("failed to execute image tool: resize requires height")
                    })?;
                    let resized = image.resize_exact(width, height, filter);
                    let format = requested_format
                        .or(detected_format)
                        .unwrap_or(ImageFormat::Png);
                    render_output(resized, format, quality, output_path.as_deref())
                }
                "convert" => {
                    let format = requested_format.or(detected_format).ok_or_else(|| {
                        anyhow::anyhow!("failed to execute image tool: convert requires format")
                    })?;
                    render_output(image, format, quality, output_path.as_deref())
                }
                other => anyhow::bail!("failed to execute image tool: unknown action '{other}'"),
            }
        })
        .await??;

        Ok(ToolResult::Json { value: result })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ImageFormat {
    Png,
    Jpeg,
    Webp,
    Gif,
}

impl ImageFormat {
    fn as_str(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpeg",
            Self::Webp => "webp",
            Self::Gif => "gif",
        }
    }
}

impl From<image::ImageFormat> for ImageFormat {
    fn from(value: image::ImageFormat) -> Self {
        match value {
            image::ImageFormat::Png => Self::Png,
            image::ImageFormat::Jpeg => Self::Jpeg,
            image::ImageFormat::WebP => Self::Webp,
            image::ImageFormat::Gif => Self::Gif,
            _ => Self::Png,
        }
    }
}

fn parse_format(value: &str) -> anyhow::Result<ImageFormat> {
    match value {
        "png" => Ok(ImageFormat::Png),
        "jpeg" | "jpg" => Ok(ImageFormat::Jpeg),
        "webp" => Ok(ImageFormat::Webp),
        "gif" => Ok(ImageFormat::Gif),
        other => anyhow::bail!("failed to execute image tool: unsupported format '{other}'"),
    }
}

fn infer_format_from_path(path: &Path) -> Option<ImageFormat> {
    parse_format(&path.extension()?.to_str()?.to_ascii_lowercase()).ok()
}

fn parse_filter(value: Option<&str>) -> anyhow::Result<FilterType> {
    match value.unwrap_or("lanczos3") {
        "nearest" => Ok(FilterType::Nearest),
        "triangle" => Ok(FilterType::Triangle),
        "catmull-rom" => Ok(FilterType::CatmullRom),
        "gaussian" => Ok(FilterType::Gaussian),
        "lanczos3" => Ok(FilterType::Lanczos3),
        other => anyhow::bail!("failed to execute image tool: unsupported filter '{other}'"),
    }
}

fn decode_image(bytes: &[u8]) -> anyhow::Result<DynamicImage> {
    ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()?
        .decode()
        .map_err(Into::into)
}

fn render_output(
    image: DynamicImage,
    format: ImageFormat,
    quality: u8,
    output_path: Option<&Path>,
) -> anyhow::Result<Value> {
    let encoded = encode_image(&image, format, quality)?;
    if let Some(output_path) = output_path {
        atomic_write_blocking(output_path, &encoded)?;
        Ok(json!({
            "output_path": output_path,
            "format": format.as_str(),
            "width": image.width(),
            "height": image.height(),
        }))
    } else {
        Ok(json!({
            "format": format.as_str(),
            "width": image.width(),
            "height": image.height(),
            "data_base64": STANDARD.encode(encoded),
        }))
    }
}

fn encode_image(image: &DynamicImage, format: ImageFormat, quality: u8) -> anyhow::Result<Vec<u8>> {
    let mut buffer = Vec::new();
    match format {
        ImageFormat::Png => image.write_to(&mut Cursor::new(&mut buffer), ImageCrateFormat::Png)?,
        ImageFormat::Jpeg => {
            let encoder = JpegEncoder::new_with_quality(&mut buffer, quality);
            image.write_with_encoder(encoder)?;
        }
        ImageFormat::Webp => {
            let encoder = WebPEncoder::new_lossless(&mut buffer);
            image.write_with_encoder(encoder)?;
        }
        ImageFormat::Gif => image.write_to(&mut Cursor::new(&mut buffer), ImageCrateFormat::Gif)?,
    }
    Ok(buffer)
}

#[cfg(all(test, feature = "image-tools"))]
mod tests {
    use super::*;

    #[test]
    fn resize_returns_encoded_payload() {
        let temp = tempfile::tempdir().expect("tempdir");
        let input = temp.path().join("input.png");
        let image = DynamicImage::new_rgba8(2, 2);
        let bytes = encode_image(&image, ImageFormat::Png, 80).expect("encode");
        std::fs::write(&input, bytes).expect("write");

        let value = render_output(
            image.resize_exact(1, 1, FilterType::Nearest),
            ImageFormat::Png,
            80,
            None,
        )
        .expect("render output");

        assert_eq!(value["width"], 1);
        assert!(
            value["data_base64"]
                .as_str()
                .is_some_and(|value| !value.is_empty())
        );
    }

    #[test]
    fn infer_format_from_path_uses_extension() {
        assert_eq!(
            infer_format_from_path(Path::new("out.jpg")),
            Some(ImageFormat::Jpeg)
        );
        assert_eq!(
            infer_format_from_path(Path::new("out.webp")),
            Some(ImageFormat::Webp)
        );
        assert_eq!(infer_format_from_path(Path::new("out.txt")), None);
    }
}
