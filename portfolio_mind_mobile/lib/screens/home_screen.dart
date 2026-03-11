import 'package:flutter/material.dart';
import '../services/api_service.dart';
import 'profile_screen.dart';
import 'portfolio_screen.dart';
import 'recommendation_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _selectedIndex = 0;

  final List<Widget> _pages = const [
    DashboardSection(),
    ProfileScreen(),
    PortfolioScreen(),
    RecommendationScreen(),
  ];

  final List<String> _titles = const [
    'Portfolio Mind',
    'Profil utilisateur',
    'Portefeuille',
    'Recommandation IA',
  ];

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(_titles[_selectedIndex]),
        centerTitle: true,
      ),
      body: _pages[_selectedIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: _onItemTapped,
        selectedItemColor: Colors.deepPurple,
        unselectedItemColor: Colors.grey,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.home),
            label: 'Accueil',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.person),
            label: 'Profil',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.account_balance_wallet),
            label: 'Portefeuille',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.auto_graph),
            label: 'IA',
          ),
        ],
      ),
    );
  }
}

class DashboardSection extends StatefulWidget {
  const DashboardSection({super.key});

  @override
  State<DashboardSection> createState() => _DashboardSectionState();
}

class _DashboardSectionState extends State<DashboardSection> {
  final ApiService _apiService = ApiService();

  String _userStatus = 'Pas encore testé';
  String _portfolioStatus = 'Pas encore testé';
  bool _loadingUser = false;
  bool _loadingPortfolio = false;

  Future<void> _testUserService() async {
    setState(() {
      _loadingUser = true;
    });

    try {
      final result = await _apiService.checkUserHealth();
      setState(() {
        _userStatus = result;
      });
    } catch (e) {
      setState(() {
        _userStatus = 'Erreur: $e';
      });
    } finally {
      setState(() {
        _loadingUser = false;
      });
    }
  }

  Future<void> _testPortfolioService() async {
    setState(() {
      _loadingPortfolio = true;
    });

    try {
      final result = await _apiService.checkPortfolioHealth();
      setState(() {
        _portfolioStatus = result;
      });
    } catch (e) {
      setState(() {
        _portfolioStatus = 'Erreur: $e';
      });
    } finally {
      setState(() {
        _loadingPortfolio = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Bienvenue sur Portfolio Mind',
            style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          const Text(
            'Application mobile intelligente pour la gestion et la recommandation de portefeuille.',
            style: TextStyle(fontSize: 16),
          ),
          const SizedBox(height: 24),

          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Test backend AWS',
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 16),

                  ElevatedButton.icon(
                    onPressed: _loadingUser ? null : _testUserService,
                    icon: const Icon(Icons.cloud_done),
                    label: const Text('Tester user_service'),
                  ),
                  const SizedBox(height: 8),
                  Text('Statut user_service : $_userStatus'),

                  const SizedBox(height: 24),

                  ElevatedButton.icon(
                    onPressed: _loadingPortfolio ? null : _testPortfolioService,
                    icon: const Icon(Icons.cloud_done),
                    label: const Text('Tester portfolio_service'),
                  ),
                  const SizedBox(height: 8),
                  Text('Statut portfolio_service : $_portfolioStatus'),
                ],
              ),
            ),
          ),

          const SizedBox(height: 24),

          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: const [
                  Text(
                    'État du projet',
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                  ),
                  SizedBox(height: 12),
                  Text('• Backend microservices avec FastAPI'),
                  Text('• Docker / Docker Compose'),
                  Text('• API Gateway NGINX'),
                  Text('• Kubernetes'),
                  Text('• Terraform'),
                  Text('• AWS Deployment'),
                  Text('• Mobile Flutter en cours'),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
