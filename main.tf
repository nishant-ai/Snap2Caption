# main.tf

# ------------------------------
# KVM Instance Definition
# ------------------------------
resource "openstack_compute_instance_v2" "kvm_instance" {
  provider        = openstack.kvm
  name            = "mlops-kvm-vm"
  image_name      = var.kvm_image_name
  flavor_name     = var.kvm_flavor_name
  key_pair        = var.kvm_key_name
  security_groups = ["default"]

  network {
    name = var.kvm_network_name
  }

  tags = [
    "environment:production",
    "managed_by:terraform",
    "project:mlops"
  ]
}

resource "openstack_networking_floatingip_v2" "kvm_fip" {
  provider = openstack.kvm
  pool     = "public"
}

resource "openstack_compute_floatingip_associate_v2" "kvm_fip_assoc" {
  provider    = openstack.kvm
  floating_ip = openstack_networking_floatingip_v2.kvm_fip.address
  instance_id = openstack_compute_instance_v2.kvm_instance.id
}

output "kvm_floating_ip" {
  description = "Public IP address of the KVM instance"
  value       = openstack_networking_floatingip_v2.kvm_fip.address
}

# ------------------------------
# CHI@UC Instance Definition
# ------------------------------
resource "openstack_compute_instance_v2" "chiuc_instance" {
  provider        = openstack.chiuc
  name            = "mlops-chiuc-vm"
  image_name      = var.chiuc_image_name
  flavor_name     = var.chiuc_flavor_name
  key_pair        = var.chiuc_key_name
  security_groups = ["default"]

  network {
    name = var.chiuc_network_name
  }

  tags = [
    "environment:production",
    "managed_by:terraform",
    "project:mlops"
  ]
}

resource "openstack_networking_floatingip_v2" "chiuc_fip" {
  provider = openstack.chiuc
  pool     = "public"
}

resource "openstack_compute_floatingip_associate_v2" "chiuc_fip_assoc" {
  provider    = openstack.chiuc
  floating_ip = openstack_networking_floatingip_v2.chiuc_fip.address
  instance_id = openstack_compute_instance_v2.chiuc_instance.id
}

output "chiuc_floating_ip" {
  description = "Public IP address of the CHI@UC instance"
  value       = openstack_networking_floatingip_v2.chiuc_fip.address
}
