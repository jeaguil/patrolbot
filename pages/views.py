from django.shortcuts import render, redirect


def welcome_view(request):
    return render(request, "pages/welcome.html", {})


def dashboard_view(request):
    if request.user.is_authenticated:
        return render(request, "pages/dashboard.html", {})
    else:
        return redirect("/")
