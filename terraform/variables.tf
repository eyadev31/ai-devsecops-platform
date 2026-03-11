variable "namespace" {
  type    = string
  default = "portfolio-mind"
}

variable "user_service_image" {
  type    = string
  default = "user-service:latest"
}

variable "portfolio_service_image" {
  type    = string
  default = "portfolio-service:latest"
}

variable "gateway_image" {
  type    = string
  default = "gateway-k8s:latest"
}

variable "user_service_port" {
  type    = number
  default = 8000
}

variable "portfolio_service_port" {
  type    = number
  default = 8001
}

variable "gateway_port" {
  type    = number
  default = 80
}


