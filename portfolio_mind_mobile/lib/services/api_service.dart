import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/user_profile.dart';
import '../models/portfolio.dart';
import '../models/recommendation.dart';

class ApiService {
  static const String baseUrl = 'http://35.175.177.80:8080';

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

  Future<UserProfile> getUserProfile(int userId) async {
    final response = await http.get(Uri.parse('$baseUrl/users/profile/$userId'));
    if (response.statusCode == 200) {
      return UserProfile.fromJson(jsonDecode(response.body));
    } else {
      throw Exception('Impossible de charger le profil');
    }
  }

  Future<Portfolio> getPortfolio(int userId) async {
    final response = await http.get(Uri.parse('$baseUrl/portfolio/user/$userId'));
    if (response.statusCode == 200) {
      return Portfolio.fromJson(jsonDecode(response.body));
    } else {
      throw Exception('Impossible de charger le portefeuille');
    }
  }

  Future<Recommendation> getRecommendation(int userId) async {
    final response = await http.get(Uri.parse('$baseUrl/portfolio/recommendation/$userId'));
    if (response.statusCode == 200) {
      return Recommendation.fromJson(jsonDecode(response.body));
    } else {
      throw Exception('Impossible de charger la recommandation');
    }
  }

  Future<void> createUserProfile(UserProfile profile) async {
    final response = await http.post(
      Uri.parse('$baseUrl/users/profile'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(profile.toJson()),
    );

    if (response.statusCode != 200) {
      throw Exception('Erreur lors de la création du profil');
    }
  }
}
