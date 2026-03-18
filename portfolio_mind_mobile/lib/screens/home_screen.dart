import 'package:flutter/material.dart';

import '../services/api_service.dart';
import '../theme/app_theme.dart';
import '../utils/app_constants.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ApiService _apiService = ApiService();

  String _userServiceStatus = 'Pas encore testé';
  String _portfolioServiceStatus = 'Pas encore testé';

  bool _isTestingUser = false;
  bool _isTestingPortfolio = false;

  Future<void> _testUserService() async {
    setState(() => _isTestingUser = true);
    try {
      final result = await _apiService.testUserService();
      setState(() {
        _userServiceStatus = result;
      });
    } catch (e) {
      setState(() {
        _userServiceStatus = 'Erreur: $e';
      });
    } finally {
      setState(() => _isTestingUser = false);
    }
  }

  Future<void> _testPortfolioService() async {
    setState(() => _isTestingPortfolio = true);
    try {
      final result = await _apiService.testPortfolioService();
      setState(() {
        _portfolioServiceStatus = result;
      });
    } catch (e) {
      setState(() {
        _portfolioServiceStatus = 'Erreur: $e';
      });
    } finally {
      setState(() => _isTestingPortfolio = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text(AppConstants.appName),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(20, 18, 20, 100),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildHeroSection(context),
              const SizedBox(height: 20),
              _buildSummaryRow(context),
              const SizedBox(height: 20),
              _buildBackendCard(context),
              const SizedBox(height: 20),
              Text(
                'Vue d’ensemble',
                style: textTheme.headlineSmall,
              ),
              const SizedBox(height: 12),
              _buildOverviewCard(
                icon: Icons.account_balance_wallet_rounded,
                title: 'Résumé portefeuille',
                subtitle:
                    'Allocation équilibrée avec exposition en actions, ETF, crypto, cash et or.',
              ),
              _buildOverviewCard(
                icon: Icons.auto_awesome_rounded,
                title: 'Résumé IA',
                subtitle:
                    'Le système recommande un portefeuille diversifié cohérent avec un profil modéré.',
              ),
              _buildOverviewCard(
                icon: Icons.cloud_done_rounded,
                title: 'Infrastructure cloud',
                subtitle:
                    'Backend déployé sur AWS EC2 avec FastAPI, Docker Compose et NGINX Gateway.',
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeroSection(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(22),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(28),
        gradient: const LinearGradient(
          colors: [
            Color(0xFF6C4DFF),
            Color(0xFF8B72FF),
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        boxShadow: const [
          BoxShadow(
            color: Color(0x226C4DFF),
            blurRadius: 18,
            offset: Offset(0, 10),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(
            Icons.auto_graph_rounded,
            size: 36,
            color: Colors.white,
          ),
          const SizedBox(height: 18),
          Text(
            AppConstants.homeTitle,
            style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                  color: Colors.white,
                  fontSize: 32,
                ),
          ),
          const SizedBox(height: 12),
          Text(
            AppConstants.appTagline,
            style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                  color: Colors.white.withOpacity(0.92),
                ),
          ),
          const SizedBox(height: 20),
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: const [
              _Badge(text: 'Cloud Native'),
              _Badge(text: 'FastAPI'),
              _Badge(text: 'Flutter'),
              _Badge(text: 'AI Agents'),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildSummaryRow(BuildContext context) {
    return Row(
      children: const [
        Expanded(
          child: _StatCard(
            title: 'Profil',
            value: 'Modéré',
            icon: Icons.shield_moon_rounded,
          ),
        ),
        SizedBox(width: 12),
        Expanded(
          child: _StatCard(
            title: 'Portefeuille',
            value: '5 actifs',
            icon: Icons.pie_chart_rounded,
          ),
        ),
      ],
    );
  }

  Widget _buildBackendCard(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              AppConstants.backendTestTitle,
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 14),
            _buildServiceTestTile(
              context: context,
              title: 'user_service',
              status: _userServiceStatus,
              loading: _isTestingUser,
              icon: Icons.cloud_done_rounded,
              onPressed: _testUserService,
            ),
            const SizedBox(height: 12),
            _buildServiceTestTile(
              context: context,
              title: 'portfolio_service',
              status: _portfolioServiceStatus,
              loading: _isTestingPortfolio,
              icon: Icons.account_balance_wallet_rounded,
              onPressed: _testPortfolioService,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildServiceTestTile({
    required BuildContext context,
    required String title,
    required String status,
    required bool loading,
    required IconData icon,
    required VoidCallback onPressed,
  }) {
    final bool isSuccess = status.toLowerCase().contains('running');

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.6),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFFE6E1EF)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              CircleAvatar(
                radius: 20,
                backgroundColor: const Color(0xFFEDE7FA),
                child: Icon(icon, color: AppTheme.primaryColor),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  title,
                  style: Theme.of(context).textTheme.titleMedium,
                ),
              ),
              ElevatedButton(
                onPressed: loading ? null : onPressed,
                child: loading
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Colors.white,
                        ),
                      )
                    : const Text('Tester'),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            'Statut : $status',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: isSuccess ? AppTheme.accentColor : AppTheme.textSecondary,
                  fontWeight: isSuccess ? FontWeight.w600 : FontWeight.w500,
                ),
          ),
        ],
      ),
    );
  }

  Widget _buildOverviewCard({
    required IconData icon,
    required String title,
    required String subtitle,
  }) {
    return Card(
      child: ListTile(
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        leading: CircleAvatar(
          backgroundColor: const Color(0xFFEDE7FA),
          child: Icon(icon, color: AppTheme.primaryColor),
        ),
        title: Text(
          title,
          style: const TextStyle(fontWeight: FontWeight.w600),
        ),
        subtitle: Padding(
          padding: const EdgeInsets.only(top: 6),
          child: Text(subtitle),
        ),
      ),
    );
  }
}

class _StatCard extends StatelessWidget {
  final String title;
  final String value;
  final IconData icon;

  const _StatCard({
    required this.title,
    required this.value,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 18),
        child: Row(
          children: [
            CircleAvatar(
              backgroundColor: const Color(0xFFEDE7FA),
              child: Icon(icon, color: AppTheme.primaryColor),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    value,
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _Badge extends StatelessWidget {
  final String text;

  const _Badge({required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.18),
        borderRadius: BorderRadius.circular(30),
        border: Border.all(color: Colors.white.withOpacity(0.22)),
      ),
      child: Text(
        text,
        style: const TextStyle(
          color: Colors.white,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }
}
