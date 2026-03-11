output "namespace" {
  value = kubernetes_namespace.portfolio_mind.metadata[0].name
}

output "user_service_name" {
  value = kubernetes_service.user_service.metadata[0].name
}

output "portfolio_service_name" {
  value = kubernetes_service.portfolio_service.metadata[0].name
}

output "gateway_service_name" {
  value = kubernetes_service.gateway.metadata[0].name
}
