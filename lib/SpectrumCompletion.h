#pragma once

#include "Config.h"
#include "SpectrumData.h"

#include <string>

namespace trspv {

/// Interpolation and completion module.
class SpectrumCompletion {
public:
    /**
     * Complete the original spectrum data according to the interpolation config.
     * data: input spectrum samples.
     * config: interpolation/completion settings.
     * Returns the completed spectrum.
     */
    static SpectrumData complete(const SpectrumData& data,
        const SpectrumCompletionConfig& config);
};

} // namespace trspv
