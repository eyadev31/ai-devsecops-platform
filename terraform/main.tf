resource "kubernetes_namespace" "portfolio_mind" {
  metadata {
    name = var.namespace
  }
}

resource "kubernetes_deployment" "user_service" {
  metadata {
    name      = "user-service"
    namespace = kubernetes_namespace.portfolio_mind.metadata[0].name
    labels = {
      app = "user-service"
    }
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "user-service"
      }
    }

    template {
      metadata {
        labels = {
          app = "user-service"
        }
      }

      spec {
        container {
          name  = "user-service"
          image = var.user_service_image

          port {
            container_port = var.user_service_port
          }

          image_pull_policy = "Never"
        }
      }
    }
  }
}

resource "kubernetes_service" "user_service" {
  metadata {
    name      = "user-service"
    namespace = kubernetes_namespace.portfolio_mind.metadata[0].name
  }

  spec {
    selector = {
      app = "user-service"
    }

    port {
      port        = 8000
      target_port = var.user_service_port
    }

    type = "ClusterIP"
  }
}

resource "kubernetes_deployment" "portfolio_service" {
  metadata {
    name      = "portfolio-service"
    namespace = kubernetes_namespace.portfolio_mind.metadata[0].name
    labels = {
      app = "portfolio-service"
    }
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "portfolio-service"
      }
    }

    template {
      metadata {
        labels = {
          app = "portfolio-service"
        }
      }

      spec {
        container {
          name  = "portfolio-service"
          image = var.portfolio_service_image

          port {
            container_port = var.portfolio_service_port
          }

          image_pull_policy = "Never"
        }
      }
    }
  }
}

resource "kubernetes_service" "portfolio_service" {
  metadata {
    name      = "portfolio-service"
    namespace = kubernetes_namespace.portfolio_mind.metadata[0].name
  }

  spec {
    selector = {
      app = "portfolio-service"
    }

    port {
      port        = 8001
      target_port = var.portfolio_service_port
    }

    type = "ClusterIP"
  }
}

resource "kubernetes_deployment" "gateway" {
  metadata {
    name      = "gateway"
    namespace = kubernetes_namespace.portfolio_mind.metadata[0].name
    labels = {
      app = "gateway"
    }
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "gateway"
      }
    }

    template {
      metadata {
        labels = {
          app = "gateway"
        }
      }

      spec {
        container {
          name  = "gateway"
          image = var.gateway_image

          port {
            container_port = var.gateway_port
          }

          image_pull_policy = "Never"
        }
      }
    }
  }
}

resource "kubernetes_service" "gateway" {
  metadata {
    name      = "gateway"
    namespace = kubernetes_namespace.portfolio_mind.metadata[0].name
  }

  spec {
    selector = {
      app = "gateway"
    }

    port {
      port        = 80
      target_port = var.gateway_port
    }

    type = "ClusterIP"
  }
}

