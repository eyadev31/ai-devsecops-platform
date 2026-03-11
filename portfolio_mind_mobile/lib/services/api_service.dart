import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = 'http://100.26.23.4:8080';

  Future<String> checkUserHealth() async {
    final response = await http.get(Uri.parse('$baseUrl/users/health'));

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['status'] ?? 'Réponse user_service reçue';
    } else {
      throw Exception('Erreur user_service: ${response.statusCode}');
    }
  }

  Future<String> checkPortfolioHealth() async {
    final response = await http.get(Uri.parse('$baseUrl/portfolio/health'));

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['status'] ?? 'Réponse portfolio_service reçue';
    } else {
      throw Exception('Erreur portfolio_service: ${response.statusCode}');
    }
  }
}
