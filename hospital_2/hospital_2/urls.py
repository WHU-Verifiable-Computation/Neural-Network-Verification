"""hosptial URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.views.decorators.csrf import csrf_exempt

from hosptialapp import views
from django.views.generic.base import TemplateView

urlpatterns = [
    path('getcsrf',views.getcsrf),
    path('',TemplateView.as_view(template_name="index.html")),
    path('admin/', admin.site.urls),
    # path('',views.chooselogin),
    path('doctorlogin/',views.docterlogin),
    path('adminlogin/',views.adminlogin),
    path('register/',views.register),
    path('login/',views.login),
    path('addrecord/',views.addrecord),
    path('info/',views.info),
    path('logout/',views.logout),
    path('roles/',views.getrole),
    path('record_create/',views.addrecord1),
    path('record_all/',views.get_all_record),
    path('fetchrecord_id/',views.get_record_id),
    path('update_record/',views.update_record),
    path('verify_id/',views.verify_id),
    path('profile_update/',views.update),
    path('patient_list/',views.patient_list),
    path('update_status/',views.update_status),
    path('updated_bypatient/',views.update_bypatient),
    path('docter_list/',views.docter_list),
    path('updated_byhospital/',views.update_byhospital),
    path('add_docter/',views.add_docter),
    path('delete_docter/',views.delete_docter),
    path('storage_verification/',views.storage_verification),
    path('verify_batch/',views.verify_batch),
    path('destroy/',views.messyBlocks),
    path('exportpdf/',views.exportpdf),
    path('log_list/',views.loglist),
    path('result_write/',views.result_write),
    path('edit_net/',views.edit_net),
    path('reset_net/',views.reset_net),
    path('verify2/',views.verify2),
    path('destory_location/',views.destroy_location),
    path('delete_record/',views.handledelete),
    path('multiattri_search/',views.handleFilter_multiattri)
]