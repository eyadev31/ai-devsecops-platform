import 'package:flutter/material.dart';
import '../services/api_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ApiService _apiService = ApiService();

  String _userStatus = 'Pas encore testé';
  String _portfolioStatus = 'Pas encore testé';
  bool _loadingUser = false;
  bool _loadingPortfolio = false;

  Future<void> _testUserService() async {
    setState(() => _loadingUser = true);
    try {
      final status = await _apiService.checkUserHealth();
      setState(() => _userStatus = status);
    } catch (e) {
      setState(() => _userStatus = 'Erreur: $e');
    } finally {
      setState(() => _loadingUser = false);
    }
  }

  Future<void> _testPortfolioService() async {
    setState(() => _loadingPortfolio = true);
    try {
      final status = await _apiService.checkPortfolioHealth();
      setState(() => _portfolioStatus = status);
    } catch (e) {
      setState(() => _portfolioStatus = 'Erreur: $e');
    } finally {
      setState(() => _loadingPortfolio = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        const Text(
          'Bienvenue sur Portfolio Mind',
          style: TextStyle(fontSize: 30, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 10),
        const Text(
          'Application intelligente pour la gestion et la recommandation de portefeuille.',
          style: TextStyle(fontSize: 16, color: Colors.black87),
        ),
        const SizedBox(height: 24),
        Card(
          elevation: 2,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Test backend AWS',
                  style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 16),
                ElevatedButton.icon(
                  onPressed: _loadingUser ? null : _testUserService,
                  icon: _loadingUser
                      ? const SizedBox(
                          height: 18,
                          width: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.cloud_done),
                  label: const Text('Tester user_service'),
                ),
                const SizedBox(height: 8),
                Text('Statut user_service : $_userStatus'),
                const SizedBox(height: 20),
                ElevatedButton.icon(
                  onPressed: _loadingPortfolio ? null : _testPortfolioService,
                  icon: _loadingPortfolio
                      ? const SizedBox(
                          height: 18,
                          width: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.account_balance_wallet),
                  label: const Text('Tester portfolio_service'),
                ),
                const SizedBox(height: 8),
                Text('Statut portfolio_service : $_portfolioStatus'),
              ],
            ),
          ),
        ),
        const SizedBox(height: 20),
        Card(
          elevation: 2,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
          child: const Padding(
            padding: EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'État du projet',
                  style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 12),
                Text('• Backend microservices avec FastAPI'),
                Text('• Déploiement Docker / Docker Compose'),
                Text('• API Gateway NGINX'),
                Text('• Déploiement AWS EC2'),
                Text('• Application Flutter connectée au backend'),
                Text('• Recommandation IA dynamique'),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
