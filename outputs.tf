# outputs.tf

# ------------------------------
# KVM Instance Outputs
# ------------------------------
output "kvm_instance_ip" {
  description = "Floating IP address of the KVM instance"
  value       = openstack_networking_floatingip_v2.kvm_fip.address
}

output "kvm_instance_name" {
  description = "Name of the KVM instance"
  value       = openstack_compute_instance_v2.kvm_instance.name
}

output "kvm_instance_provider" {
  description = "Provider for the KVM instance"
  value       = "KVM@TACC"
}

output "kvm_instance_details" {
  description = "Details of the KVM instance"
  value = {
    ip       = openstack_networking_floatingip_v2.kvm_fip.address
    instance = openstack_compute_instance_v2.kvm_instance.name
    provider = "KVM@TACC"
  }
}

# ------------------------------
# CHI@UC Instance Outputs
# ------------------------------
output "chiuc_instance_ip" {
  description = "Floating IP address of the CHI@UC instance"
  value       = openstack_networking_floatingip_v2.chiuc_fip.address
}

output "chiuc_instance_name" {
  description = "Name of the CHI@UC instance"
  value       = openstack_compute_instance_v2.chiuc_instance.name
}

output "chiuc_instance_provider" {
  description = "Provider for the CHI@UC instance"
  value       = "CHI@UC"
}

output "chiuc_instance_details" {
  description = "Details of the CHI@UC instance"
  value = {
    ip       = openstack_networking_floatingip_v2.chiuc_fip.address
    instance = openstack_compute_instance_v2.chiuc_instance.name
    provider = "CHI@UC"
  }
}
