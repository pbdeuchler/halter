use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "software-factory")]
#[command(about = "Example Halter workflow that turns GitHub issues into an implementation PR")]
pub(crate) struct Cli {
    #[arg(
        long,
        default_value = "origin",
        help = "Git remote whose GitHub URL identifies the repository"
    )]
    pub(crate) remote: String,
    #[arg(long, help = "Base branch; defaults to the repository default branch")]
    pub(crate) base: Option<String>,
    #[arg(
        long,
        help = "Branch name to create; defaults to a generated factory branch"
    )]
    pub(crate) branch: Option<String>,
    #[arg(
        long,
        help = "Create and run inside a dedicated git worktree under /tmp"
    )]
    pub(crate) worktree: bool,
    #[arg(
        long,
        help = "Poll the opened PR for reviews and /plsfix comments until it merges"
    )]
    pub(crate) monitor: bool,
    #[arg(long, help = "Allow starting from a dirty worktree")]
    pub(crate) allow_dirty: bool,
    #[arg(
        long,
        help = "Include the generated implementation plan file in commits"
    )]
    pub(crate) commit_impl_plan: bool,
    #[arg(
        long,
        conflicts_with = "reset_checkpoint",
        help = "Resume from the factory checkpoint file for this worktree"
    )]
    pub(crate) resume: bool,
    #[arg(
        long,
        help = "Delete any existing factory checkpoint before starting a fresh run"
    )]
    pub(crate) reset_checkpoint: bool,
    #[arg(long, help = "Work on one specific open GitHub issue number")]
    pub(crate) issue: Option<u64>,
    #[arg(
        long,
        default_value_t = 5,
        help = "Maximum Kimi/GPT review-repair iterations"
    )]
    pub(crate) max_review_iterations: usize,
    #[arg(long, default_value_t = 60, help = "Seconds between PR monitor polls")]
    pub(crate) poll_seconds: u64,
    #[arg(
        long,
        default_value = "openrouter/z-ai/glm-5.2",
        help = "Provider/model for issue grouping and /plsfix refinement"
    )]
    pub(crate) glm_model: String,
    #[arg(
        long,
        default_value = "openrouter/moonshotai/kimi-k2.7-code",
        help = "Provider/model for implementation"
    )]
    pub(crate) implementer_model: String,
    #[arg(
        long,
        default_value = "openrouter/z-ai/glm-5.2",
        help = "Provider/model for branch-diff code review"
    )]
    pub(crate) reviewer_model: String,
    #[arg(
        long,
        default_value = "openrouter/google/gemma-4-31b-it",
        help = "Provider/model for PR title and body drafting"
    )]
    pub(crate) pr_model: String,
}
