#!/usr/bin/env bash
# =============================================================================
# cross-platform.sh
# Shared cross-platform helpers for Mac (launchd) and Linux (systemd).
# Source this file before any platform-specific operations.
# =============================================================================

# --- OS Detection -----------------------------------------------------------
_cp_os=""
cp_detect_os() {
    if [[ -z "${_cp_os}" ]]; then
        _cp_os="$(uname -s)"
    fi
}

is_mac() {
    cp_detect_os
    [[ "${_cp_os}" == "Darwin" ]]
}

is_linux() {
    cp_detect_os
    [[ "${_cp_os}" == "Linux" ]]
}

# --- Portable stat ----------------------------------------------------------
cp_stat_user() {
    local file="$1"
    if is_mac; then
        stat -f '%Su' "${file}"
    else
        stat -c '%U' "${file}"
    fi
}

cp_stat_group() {
    local file="$1"
    if is_mac; then
        stat -f '%Sg' "${file}"
    else
        stat -c '%G' "${file}"
    fi
}

# --- Portable sed -i --------------------------------------------------------
cp_sed_inplace() {
    if is_mac; then
        sed -i '' "$@"
    else
        sed -i "$@"
    fi
}

# --- Portable readlink -f ---------------------------------------------------
cp_readlink() {
    local path="$1"
    if is_mac; then
        python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "${path}"
    else
        readlink -f "${path}"
    fi
}

# --- Portable get home directory --------------------------------------------
cp_get_home_dir() {
    local user="$1"
    if is_mac; then
        dscl . -read "/Users/${user}" NFSHomeDirectory 2>/dev/null | awk '{print $2}'
    else
        getent passwd "${user}" | cut -d: -f6
    fi
}

# --- Service management wrappers --------------------------------------------
# On Linux these call systemctl; on macOS they call launchctl with
# plists in /Library/LaunchDaemons/com.llmstack.<name>.plist

_svc_plist_dir="/Library/LaunchDaemons"
_svc_label_prefix="com.llmstack"

svc_plist_path() {
    echo "${_svc_plist_dir}/${_svc_label_prefix}.${1}.plist"
}

svc_label() {
    echo "${_svc_label_prefix}.${1}"
}

svc_daemon_reload() {
    if is_linux; then
        systemctl daemon-reload 2>/dev/null || true
    fi
    # macOS launchd does not need a reload
}

svc_start() {
    local name="$1"
    if is_linux; then
        systemctl start "${name}"
    else
        local label
        label="$(svc_label "${name}")"
        launchctl bootout "system/${label}" 2>/dev/null || true
        launchctl bootstrap system "$(svc_plist_path "${name}")"
    fi
}

svc_stop() {
    local name="$1"
    if is_linux; then
        systemctl stop "${name}" || true
    else
        local label
        label="$(svc_label "${name}")"
        launchctl bootout "system/${label}" 2>/dev/null || true
    fi
}

svc_restart() {
    local name="$1"
    if is_linux; then
        systemctl restart "${name}"
    else
        svc_stop "${name}"
        svc_start "${name}"
    fi
}

svc_enable() {
    local name="$1"
    if is_linux; then
        systemctl enable "${name}"
    else
        local plist
        plist="$(svc_plist_path "${name}")"
        if [[ -f "${plist}" ]]; then
            launchctl bootstrap system "${plist}"
        fi
    fi
}

svc_disable() {
    local name="$1"
    if is_linux; then
        systemctl disable "${name}" 2>/dev/null || true
    else
        local label
        label="$(svc_label "${name}")"
        launchctl bootout "system/${label}" 2>/dev/null || true
    fi
}

svc_is_active() {
    local name="$1"
    if is_linux; then
        systemctl is-active --quiet "${name}" 2>/dev/null
    else
        local label
        label="$(svc_label "${name}")"
        local plist
        plist="$(svc_plist_path "${name}")"

        # Check plist exists and is loaded
        if [[ ! -f "${plist}" ]]; then
            return 1
        fi

        # Check if the service has a PID (is running)
        local pid
        pid="$(launchctl list "${label}" 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.loads(sys.stdin.read().strip())
    print(data.get('PID', 0))
except:
    print(0)
" 2>/dev/null || echo 0)"

        [[ "${pid}" -gt 0 ]] 2>/dev/null
    fi
}

svc_is_enabled() {
    local name="$1"
    if is_linux; then
        local state
        state="$(systemctl is-enabled "${name}" 2>/dev/null || true)"
        [[ "${state}" == "enabled" || "${state}" == "enabled-runtime" ]]
    else
        local plist
        plist="$(svc_plist_path "${name}")"
        [[ -f "${plist}" ]]
    fi
}

svc_status() {
    local name="$1"
    if is_linux; then
        systemctl status --no-pager --lines=0 "${name}" 2>/dev/null || true
    else
        local label
        label="$(svc_label "${name}")"
        local plist
        plist="$(svc_plist_path "${name}")"

        echo "Service: ${name}"
        echo "Label:   ${label}"
        echo "Plist:   ${plist}"

        if [[ ! -f "${plist}" ]]; then
            echo "Status:  not installed"
            return 1
        fi

        local pid
        pid="$(launchctl list "${label}" 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.loads(sys.stdin.read().strip())
    print(data.get('PID', 0))
except:
    print(0)
" 2>/dev/null || echo 0)"

        if [[ "${pid}" -gt 0 ]] 2>/dev/null; then
            echo "Status:  active (running, PID ${pid})"
        else
            echo "Status:  inactive"
        fi
    fi
}

svc_status_all() {
    local services=("$@")
    if is_linux; then
        systemctl status --no-pager --lines=0 "${services[@]}" 2>/dev/null || true
    else
        echo "=== LLM Stack Services (macOS launchd) ==="
        echo ""
        for name in "${services[@]}"; do
            local label
            label="$(svc_label "${name}")"
            local plist
            plist="$(svc_plist_path "${name}")"

            if [[ ! -f "${plist}" ]]; then
                printf "  %-25s not installed\n" "${name}"
                continue
            fi

            local pid
            pid="$(launchctl list "${label}" 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.loads(sys.stdin.read().strip())
    print(data.get('PID', 0))
except:
    print(0)
" 2>/dev/null || echo 0)"

            if [[ "${pid}" -gt 0 ]] 2>/dev/null; then
                printf "  %-25s active (PID %s)\n" "${name}" "${pid}"
            else
                printf "  %-25s inactive\n" "${name}"
            fi
        done
    fi
}

# --- Generate launchd plist + wrapper ---------------------------------------
# generate_launchd_plist <service_name> <description> <script_filename> <timeout>
# Creates:
#   /Library/LaunchDaemons/com.llmstack.<name>.plist
#   ${STACK_DIR}/scripts/launchd-wrapper-<name>.sh
generate_launchd_plist() {
    local unit_name="$1"
    local description="$2"
    local script="$3"
    local timeout="${4:-300}"
    local label
    label="$(svc_label "${unit_name}")"
    local plist_path
    plist_path="$(svc_plist_path "${unit_name}")"
    local wrapper_path="${STACK_DIR}/scripts/launchd-wrapper-${unit_name}.sh"
    local log_file="${STACK_DIR}/logs/${unit_name}.stdout.log"
    local err_file="${STACK_DIR}/logs/${unit_name}.stderr.log"

    # Create wrapper script that sources env then execs the real script
    cat > "${wrapper_path}" <<WRAPPER
#!/usr/bin/env bash
set -euo pipefail
STACK_DIR="$(cp_readlink "${STACK_DIR}")"
set -a
source "\${STACK_DIR}/config/llm-stack.env"
set +a
exec "\${STACK_DIR}/scripts/${script}"
WRAPPER
    chmod 755 "${wrapper_path}"

    # Create plist
    cat > "${plist_path}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${wrapper_path}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${STACK_DIR}</string>
    <key>RunAtLoad</key>
    <false/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>5</integer>
    <key>StandardOutPath</key>
    <string>${log_file}</string>
    <key>StandardErrorPath</key>
    <string>${err_file}</string>
    <key>SoftResourceLimits</key>
    <dict>
        <key>NumberOfFiles</key>
        <integer>65536</integer>
    </dict>
PLIST

    # Add User/Group if provided
    if [[ -n "${SERVICE_USER:-}" ]]; then
        cat >> "${plist_path}" <<PLIST
    <key>UserName</key>
    <string>${SERVICE_USER}</string>
PLIST
    fi
    if [[ -n "${SERVICE_GROUP:-}" ]]; then
        cat >> "${plist_path}" <<PLIST
    <key>GroupName</key>
    <string>${SERVICE_GROUP}</string>
PLIST
    fi

    # systemd's After= and Conflicts= have no launchd equivalent.
    #
    # This function used to emit <key>WaitFor</key> and <key>Unbootstraps</key>
    # here. Neither is a real launchd key. launchd ignores keys it does not
    # recognise without complaint, so the units looked ordered and were not --
    # the failure was invisible for as long as nobody checked the plist against
    # Apple's documented key list.
    #
    # Recording the intent as an XML comment keeps the information for whoever
    # implements ordering properly (launchd's own answer is a dependency-aware
    # wrapper, or KeepAlive with an OtherJobEnabled condition), and saying it out
    # loud keeps it from being rediscovered the hard way a second time.
    if [[ -n "${LAUNCHD_WAIT_FOR:-}" ]]; then
        echo "    <!-- intended After=: ${LAUNCHD_WAIT_FOR} (not enforced by launchd) -->" >> "${plist_path}"
        echo "  WARNING: ${unit_name} should start after '${LAUNCHD_WAIT_FOR}'; launchd does not enforce ordering" >&2
    fi
    if [[ -n "${LAUNCHD_CONFLICTS:-}" ]]; then
        echo "    <!-- intended Conflicts=: ${LAUNCHD_CONFLICTS} (not enforced by launchd) -->" >> "${plist_path}"
        echo "  WARNING: ${unit_name} conflicts with '${LAUNCHD_CONFLICTS}'; launchd will not stop it for you" >&2
    fi

    cat >> "${plist_path}" <<PLIST
</dict>
</plist>
PLIST

    chmod 644 "${plist_path}"
    echo "  installed: ${label}.plist"
}

# --- Reset launchd plist variables ------------------------------------------
_launchd_reset() {
    unset LAUNCHD_WAIT_FOR
    unset LAUNCHD_CONFLICTS
}

# Run OS detection on source
cp_detect_os
