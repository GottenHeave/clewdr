use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct UsageBreakdown {
    #[serde(default)]
    pub total_input_tokens: u64,
    #[serde(default)]
    pub total_output_tokens: u64,
    #[serde(default)]
    pub sonnet_input_tokens: u64,
    #[serde(default)]
    pub sonnet_output_tokens: u64,
    #[serde(default)]
    pub opus_input_tokens: u64,
    #[serde(default)]
    pub opus_output_tokens: u64,
}

impl UsageBreakdown {
    pub fn any_nonzero(&self) -> bool {
        self.total_input_tokens > 0
            || self.total_output_tokens > 0
            || self.sonnet_input_tokens > 0
            || self.sonnet_output_tokens > 0
            || self.opus_input_tokens > 0
            || self.opus_output_tokens > 0
    }
}
