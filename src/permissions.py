"""
Layer 2 — Permissions Engine

Derive whether the requested action is permitted by the current
agent/user/context and produce the binary `permission_match` signal.
"""

from typing import Dict, Any

# Role-to-resource authorization matrix.
# True means the agent is allowed to perform the action on the resource.
# This is intentionally conservative for security-risk classification.
PERMISSION_RULES = {
    "customer_support_agent": {
        "read_record": {"customer_profile", "sales_pipeline"},
        "send_email": {"customer_profile", "employee_record"},
        "access_logs": {"audit_log"},
    },
    "finance_agent": {
        "read_record": {"financial_report", "employee_record"},
        "export_report": {"financial_report"},
        "access_logs": {"audit_log"},
    },
    "hr_onboarding_agent": {
        "read_record": {"employee_record", "customer_profile"},
        "create_user": {"employee_record"},
        "send_email": {"employee_record"},
    },
    "it_ops_agent": {
        "modify_config": {"system_config"},
        "access_logs": {"audit_log"},
        "delete_record": {"employee_record", "system_config"},
    },
    "sales_agent": {
        "read_record": {"customer_profile", "sales_pipeline"},
        "send_email": {"customer_profile"},
        "export_report": {"sales_pipeline", "financial_report"},
    },
    "data_analyst_agent": {
        "read_record": {"financial_report", "sales_pipeline", "customer_profile"},
        "export_report": {"financial_report", "sales_pipeline"},
        "access_logs": {"audit_log"},
    },
}

# User role overrides and additional restrictions
USER_RESTRICTIONS = {
    "vendor": {
        "denied_actions": {"modify_config", "delete_record", "create_user"},
        "allowed_resources": {"customer_profile"},
    },
    "auditor": {
        "denied_actions": set(),
        "allowed_resources": {"audit_log", "financial_report"},
    },
    "admin": {
        "denied_actions": set(),
        "allowed_resources": None,
    },
    "manager": {
        "denied_actions": {"delete_record", "modify_config"},
        "allowed_resources": None,
    },
    "developer": {
        "denied_actions": {"delete_record"},
        "allowed_resources": None,
    },
    "analyst": {
        "denied_actions": {"modify_config", "create_user"},
        "allowed_resources": None,
    },
}


def permission_match(action_context: Dict[str, Any]) -> Dict[str, int]:
    """Return a binary permission match signal for the requested action."""
    agent_role = action_context.get("agent_role")
    user_role = action_context.get("user_role")
    requested_action = action_context.get("requested_action")
    resource_type = action_context.get("resource_type")

    if not all([agent_role, user_role, requested_action, resource_type]):
        return {"permission_match": 0}

    agent_rules = PERMISSION_RULES.get(agent_role, {})
    allowed_resources = agent_rules.get(requested_action, set())
    if resource_type not in allowed_resources:
        return {"permission_match": 0}

    user_rule = USER_RESTRICTIONS.get(user_role, {})
    denied_actions = user_rule.get("denied_actions") or set()
    if requested_action in denied_actions:
        return {"permission_match": 0}

    allowed_resources_override = user_rule.get("allowed_resources")
    if allowed_resources_override is not None:
        if resource_type not in allowed_resources_override:
            return {"permission_match": 0}

    return {"permission_match": 1}
