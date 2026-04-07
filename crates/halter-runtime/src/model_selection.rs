// pattern: Functional Core

use halter_protocol::ModelId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SelectedModels {
    pub default_model: ModelId,
    pub subagent_model: ModelId,
}

pub(crate) fn select_models(
    session_default_model: &ModelId,
    session_subagent_model: &ModelId,
    turn_default_model: Option<&ModelId>,
    turn_subagent_model: Option<&ModelId>,
) -> SelectedModels {
    SelectedModels {
        default_model: turn_default_model
            .cloned()
            .unwrap_or_else(|| session_default_model.clone()),
        subagent_model: turn_subagent_model
            .cloned()
            .unwrap_or_else(|| session_subagent_model.clone()),
    }
}

#[cfg(test)]
mod tests {
    use halter_protocol::ModelId;

    use super::*;

    #[test]
    fn select_models_prefers_turn_overrides() {
        struct TestCase {
            name: &'static str,
            session_default_model: &'static str,
            session_subagent_model: &'static str,
            turn_default_model: Option<&'static str>,
            turn_subagent_model: Option<&'static str>,
            want_default_model: &'static str,
            want_subagent_model: &'static str,
        }

        let cases = [
            TestCase {
                name: "falls back to session models",
                session_default_model: "default",
                session_subagent_model: "subagent",
                turn_default_model: None,
                turn_subagent_model: None,
                want_default_model: "default",
                want_subagent_model: "subagent",
            },
            TestCase {
                name: "overrides default model only",
                session_default_model: "default",
                session_subagent_model: "subagent",
                turn_default_model: Some("subagent"),
                turn_subagent_model: None,
                want_default_model: "subagent",
                want_subagent_model: "subagent",
            },
            TestCase {
                name: "overrides subagent model only",
                session_default_model: "default",
                session_subagent_model: "subagent",
                turn_default_model: None,
                turn_subagent_model: Some("default"),
                want_default_model: "default",
                want_subagent_model: "default",
            },
            TestCase {
                name: "overrides both models",
                session_default_model: "default",
                session_subagent_model: "subagent",
                turn_default_model: Some("subagent"),
                turn_subagent_model: Some("default"),
                want_default_model: "subagent",
                want_subagent_model: "default",
            },
        ];

        for case in cases {
            let selected = select_models(
                &ModelId::from(case.session_default_model),
                &ModelId::from(case.session_subagent_model),
                case.turn_default_model.map(ModelId::from).as_ref(),
                case.turn_subagent_model.map(ModelId::from).as_ref(),
            );

            assert_eq!(
                selected.default_model,
                ModelId::from(case.want_default_model),
                "{}",
                case.name
            );
            assert_eq!(
                selected.subagent_model,
                ModelId::from(case.want_subagent_model),
                "{}",
                case.name
            );
        }
    }
}
