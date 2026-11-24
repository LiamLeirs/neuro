#pragma once

namespace neuro {
class GradMode {
public:
    static bool isEnabled() {
        return enabled_;
    }

    static void setEnabled(bool enabled) {
        enabled_ = enabled;
    }

private:
    static inline thread_local bool enabled_ = true;
};

class NoGradGuard {
public:
    NoGradGuard() : state_(GradMode::isEnabled()) {
        GradMode::setEnabled(false);
    }

    ~NoGradGuard() {
        GradMode::setEnabled(state_);
    }

    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

    NoGradGuard(NoGradGuard&&) = delete;
    NoGradGuard& operator=(NoGradGuard&&) = delete;

private:
    bool state_;
};
} // namespace neuro