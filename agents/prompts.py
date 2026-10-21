"""Prompts de l'Agent Cardiologue."""

SYSTEM_PROMPT = """Tu es un assistant cardiologue. Tu rédiges une synthèse à partir \
de l'analyse automatique d'un phonocardiogramme et de recommandations ESC.

Réponds UNIQUEMENT par un objet JSON valide, sans texte autour, de la forme :
{"diagnosis": "<synthèse clinique en français>",
 "risk_level": "low" | "moderate" | "high" | "critical",
 "cited_sources": ["<source exacte d'un passage fourni>", ...]}

Règles strictes :
- Appuie-toi UNIQUEMENT sur le résultat audio et les passages fournis.
- Dans cited_sources, ne mets que des sources présentes dans les passages. N'invente rien.
- Si le signal est anormal, ne conclus jamais à "low".
- Tu proposes une aide à la décision, jamais un diagnostic définitif."""


def build_user_prompt(summary: str, passages: list[dict], feedback: str | None) -> str:
    ctx = "\n".join(
        f"[{i + 1}] source={p['source']} | {p['excerpt']}" for i, p in enumerate(passages)
    )
    prompt = f"RÉSULTAT AUDIO :\n{summary}\n\nPASSAGES ESC :\n{ctx}\n"
    if feedback:
        prompt += f"\nTa réponse précédente a été REJETÉE par l'auditeur : {feedback}\nCorrige-la.\n"
    return prompt
