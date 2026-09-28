// pattern: Functional Core
//
// Progressive skill disclosure. The system prompt lists each skill by name
// and description; the runtime-owned `skill` tool reveals a body on demand.
// Its tool result is only an acknowledgement: the body joins the transcript
// as a user message after the step's tool results, so the cached prefix
// never changes when a skill loads. The index and the tool both derive from
// the live resource snapshot, so `replace_resources` adds or removes them
// together.

use halter_protocol::{
    CacheScope, Message, PromptSegment, PromptSegmentId, PromptSegmentKind, ResourceSnapshot,
    SkillDef, SkillName, ToolCapabilities, ToolConcurrency, ToolName, ToolResult, ToolSpec,
    UserMessage, Volatility,
};
use serde_json::{Value, json};

use crate::prompt::hash_prompt_text;

/// Name of the runtime-owned tool that loads a skill. Reserved: the session
/// loop resolves it against the resource snapshot, not the tool registry.
pub const SKILL_TOOL_NAME: &str = "skill";

/// Build the skill index segment from `(name, description)` pairs. It lives
/// in the skills section of the system prompt, behind its own cache
/// breakpoint, and changes only when the skill set does.
#[must_use]
pub fn skill_index_segment<'a>(
    skills: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> PromptSegment {
    let entries = skills
        .into_iter()
        .map(|(name, description)| match description {
            "" => format!("- {name}"),
            _ => format!("- {name}: {description}"),
        })
        .collect::<Vec<_>>()
        .join("\n");
    let text = format!(
        "# Skills\n\nSkills are packaged instructions for specific tasks. When a task matches a \
         skill below, call the `{SKILL_TOOL_NAME}` tool with its name before starting; its \
         instructions arrive in the next message. Only the names listed here are \
         valid.\n\n{entries}"
    );
    PromptSegment {
        id: PromptSegmentId::new(),
        content_hash: hash_prompt_text(&text),
        text,
        volatility: Volatility::SessionStable,
        cache_scope: CacheScope::PrefixCacheable,
        kind: PromptSegmentKind::Skill,
    }
}

/// The snapshot's skill index in name order, so the prefix is stable across
/// rebuilds. `None` when no skills are loaded.
pub(crate) fn skill_index(snapshot: &ResourceSnapshot) -> Option<PromptSegment> {
    let mut entries: Vec<(&str, &str)> = snapshot
        .skills
        .values()
        .map(|skill| (skill.name.as_str(), skill.description.as_str()))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    (!entries.is_empty()).then(|| skill_index_segment(entries))
}

/// Tool specs a request carries: the registered specs plus `skill` when the
/// snapshot has skills, kept in name order for a stable prefix.
#[must_use]
pub(crate) fn tool_specs(mut specs: Vec<ToolSpec>, snapshot: &ResourceSnapshot) -> Vec<ToolSpec> {
    if !snapshot.skills.is_empty() {
        specs.push(skill_tool_spec());
        specs.sort_by(|a, b| a.name.0.cmp(&b.name.0));
    }
    specs
}

/// Resolve a `skill` call into its tool result (an acknowledgement) and the
/// user message that carries the skill's instructions.
pub(crate) fn load_skill(
    snapshot: &ResourceSnapshot,
    args: &Value,
) -> anyhow::Result<(ToolResult, Message)> {
    let name = args
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("invalid tool input: missing string field 'name'"))?;
    let skill = snapshot.skills.get(&SkillName::from(name)).ok_or_else(|| {
        let mut available = snapshot
            .skills
            .keys()
            .map(|name| name.0.as_str())
            .collect::<Vec<_>>();
        available.sort_unstable();
        anyhow::anyhow!(
            "invalid tool input: unknown skill '{name}' (available: {})",
            available.join(", ")
        )
    })?;
    let text = format!("Loaded skill '{name}'. Its instructions follow in the next message.");
    Ok((ToolResult::Text { text }, skill_message(skill)))
}

fn skill_tool_spec() -> ToolSpec {
    ToolSpec {
        name: ToolName::from(SKILL_TOOL_NAME),
        description: "Load the instructions for a skill listed in the Skills section of the \
            system prompt. Call it as soon as a task matches a skill's description, before \
            starting the task."
            .to_owned(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Exact skill name from the Skills section."
                }
            },
            "required": ["name"]
        }),
        concurrency: ToolConcurrency::ReadOnly,
        capabilities: ToolCapabilities {
            mutating: false,
            requires_approval: false,
            cancellable: false,
            long_running: false,
        },
        provider_aliases: Default::default(),
    }
}

/// The body without frontmatter (the index already carries name and
/// description), tagged so it reads as injected context rather than as the
/// user speaking, and prefixed with its base directory so relative
/// references resolve.
fn skill_message(skill: &SkillDef) -> Message {
    let body = strip_frontmatter(&skill.body);
    let text = match skill.root.as_os_str().is_empty() {
        true => format!("<skill name=\"{}\">\n{body}\n</skill>", skill.name),
        false => format!(
            "<skill name=\"{}\">\nBase directory for this skill: {}\n\n{body}\n</skill>",
            skill.name,
            skill.root.display()
        ),
    };
    Message::User(UserMessage::text(text))
}

fn strip_frontmatter(body: &str) -> &str {
    body.strip_prefix("---\n")
        .and_then(|rest| rest.split_once("\n---\n"))
        .map_or(body, |(_, after)| after.trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot_with(skills: &[(&str, &str, &str)]) -> ResourceSnapshot {
        let mut snapshot = ResourceSnapshot::empty();
        for (name, root, body) in skills {
            snapshot.skills.insert(
                SkillName::from(*name),
                SkillDef {
                    name: (*name).to_owned(),
                    description: format!("{name} desc"),
                    body: (*body).to_owned(),
                    root: root.into(),
                    ..SkillDef::default()
                },
            );
        }
        snapshot
    }

    fn names(specs: &[ToolSpec]) -> Vec<&str> {
        specs.iter().map(|spec| spec.name.0.as_str()).collect()
    }

    #[test]
    fn skill_index_segment_lists_names_and_descriptions() {
        let segment = skill_index_segment([("pairs", "play nicely"), ("solo", "")]);

        assert_eq!(segment.kind, PromptSegmentKind::Skill);
        assert!(segment.text.starts_with("# Skills\n\n"));
        assert!(segment.text.contains("`skill` tool"));
        assert!(segment.text.ends_with("\n\n- pairs: play nicely\n- solo"));
    }

    #[test]
    fn skill_index_is_sorted_omits_bodies_and_is_none_without_skills() {
        let snapshot = snapshot_with(&[("zeta", "", "SECRET"), ("alpha", "", "SECRET")]);

        let segment = skill_index(&snapshot).expect("index");

        assert!(
            segment
                .text
                .ends_with("- alpha: alpha desc\n- zeta: zeta desc")
        );
        assert!(!segment.text.contains("SECRET"));
        assert!(skill_index(&ResourceSnapshot::empty()).is_none());
    }

    #[test]
    fn tool_specs_add_sorted_skill_tool_only_when_skills_exist() {
        let registered = ["shell", "read"]
            .map(|name| ToolSpec {
                name: ToolName::from(name),
                ..skill_tool_spec()
            })
            .to_vec();

        let with_skills = tool_specs(registered.clone(), &snapshot_with(&[("a", "", "b")]));
        let without = tool_specs(registered, &ResourceSnapshot::empty());

        assert_eq!(names(&with_skills), ["read", "shell", "skill"]);
        assert_eq!(names(&without), ["shell", "read"]);
    }

    #[test]
    fn load_skill_acknowledges_and_carries_body_in_a_user_message() {
        let cases = [
            (
                "frontmatter and root",
                "/skills/review",
                "---\nname: review\ndescription: d\n---\n\nDo the review.\n",
                "<skill name=\"review\">\nBase directory for this skill: /skills/review\n\nDo the review.\n</skill>",
            ),
            (
                "no frontmatter",
                "",
                "Just do it.",
                "<skill name=\"review\">\nJust do it.\n</skill>",
            ),
            (
                "unterminated frontmatter",
                "",
                "---\nname: x",
                "<skill name=\"review\">\n---\nname: x\n</skill>",
            ),
        ];
        for (label, root, body, expected) in cases {
            let snapshot = snapshot_with(&[("review", root, body)]);

            let (result, message) =
                load_skill(&snapshot, &json!({ "name": "review" })).expect(label);

            let Message::User(user) = message else {
                panic!("{label}: expected a user message");
            };
            assert_eq!(user.plain_text(), expected, "{label}");
            assert_eq!(
                result,
                ToolResult::Text {
                    text: "Loaded skill 'review'. Its instructions follow in the next message."
                        .to_owned()
                },
                "{label}"
            );
        }
    }

    #[test]
    fn load_skill_rejects_bad_input() {
        let snapshot = snapshot_with(&[("beta", "", "b"), ("alpha", "", "a")]);
        let cases = [
            (
                json!({ "name": "gamma" }),
                "unknown skill 'gamma' (available: alpha, beta)",
            ),
            (json!({}), "missing string field 'name'"),
        ];
        for (args, expected) in cases {
            let error = load_skill(&snapshot, &args).expect_err(expected);

            assert!(error.to_string().contains(expected), "{error}");
        }
    }
}
