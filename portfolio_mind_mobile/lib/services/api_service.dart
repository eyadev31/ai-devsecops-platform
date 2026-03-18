import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = 'http://98.88.54.142:8080';

  Future<String> testUserService() async {
    final response = await http.get(Uri.parse('$baseUrl/users/health'));

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['status'] ?? 'user service running';
    } else {
      throw Exception('Erreur user_service: ${response.statusCode}');
    }
  }

  Future<String> testPortfolioService() async {
    final response = await http.get(Uri.parse('$baseUrl/portfolio/health'));

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['status'] ?? 'portfolio service running';
    } else {
      throw Exception('Erreur portfolio_service: ${response.statusCode}');
    }
  }

  Future<Map<String, dynamic>> getUserProfile(int userId) async {
    final response = await http.get(
      Uri.parse('$baseUrl/users/profile/$userId'),
    );

    if (response.statusCode == 200) {
      return Map<String, dynamic>.from(jsonDecode(response.body));
    } else {
      throw Exception('Erreur profile: ${response.statusCode}');
    }
  }

  Future<Map<String, dynamic>> getUserPortfolio(int userId) async {
    final response = await http.get(
      Uri.parse('$baseUrl/portfolio/user/$userId'),
    );

    if (response.statusCode == 200) {
      return Map<String, dynamic>.from(jsonDecode(response.body));
    } else {
      throw Exception('Erreur portfolio: ${response.statusCode}');
    }
  }

  Future<Map<String, dynamic>> getRecommendation(int userId) async {
    final response = await http.get(
      Uri.parse('$baseUrl/portfolio/recommendation/$userId'),
    );

    if (response.statusCode == 200) {
      return Map<String, dynamic>.from(jsonDecode(response.body));
    } else {
      throw Exception('Erreur recommendation: ${response.statusCode}');
    }
  }

  Future<Map<String, dynamic>> createUserProfile(
    Map<String, dynamic> profileData,
  ) async {
    final response = await http.post(
      Uri.parse('$baseUrl/users/profile'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(profileData),
    );

    if (response.statusCode == 200 || response.statusCode == 201) {
      return Map<String, dynamic>.from(jsonDecode(response.body));
    } else {
      throw Exception('Erreur création profil: ${response.statusCode}');
    }
  }
}

