from django import forms


class SecurityAlertsForm(forms.Form):
    potential_threats = forms.CharField(
        label="",
        widget=forms.Textarea(
            attrs={
                "rows": 10,
                "cols": 5,
                "placeholder": "Potential threats",
                "readonly": True,
            },
        ),
    )


class ActionLogForm(forms.Form):
    action_logs = forms.CharField(
        label="",
        widget=forms.Textarea(
            attrs={
                "rows": 10,
                "cols": 5,
                "placeholder": "Model actions",
                "readonly": True,
            },
        ),
    )
