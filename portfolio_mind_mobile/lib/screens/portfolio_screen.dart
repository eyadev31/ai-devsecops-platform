import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

import '../services/api_service.dart';
import '../theme/app_theme.dart';
import '../utils/app_constants.dart';

class PortfolioScreen extends StatefulWidget {
  const PortfolioScreen({super.key});

  @override
  State<PortfolioScreen> createState() => _PortfolioScreenState();
}

class _PortfolioScreenState extends State<PortfolioScreen> {
  final ApiService _apiService = ApiService();

  Map<String, dynamic>? _portfolio;
  bool _isLoading = true;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _loadPortfolio();
  }

  Future<void> _loadPortfolio() async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      final portfolio = await _apiService.getUserPortfolio(1);
      setState(() {
        _portfolio = portfolio;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'Impossible de charger le portefeuille.\n$e';
        _isLoading = false;
      });
    }
  }

  List<Map<String, dynamic>> get _assets {
    final rawAssets = _portfolio?['assets'] as List<dynamic>? ?? [];
    return rawAssets.map((item) => Map<String, dynamic>.from(item)).toList();
  }

  double get _totalValue {
    return 125000.0;
  }

  double get _estimatedReturn {
    return 8.4;
  }

  double get _diversificationScore {
    return 8.7;
  }

  String get _riskLevel {
    return 'Modéré';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(AppConstants.portfolioTitle),
      ),
      body: SafeArea(
        child: _isLoading
            ? const Center(
                child: CircularProgressIndicator(
                  color: AppTheme.primaryColor,
                ),
              )
            : _errorMessage != null
                ? _buildErrorView()
                : _buildPortfolioView(),
      ),
    );
  }

  Widget _buildErrorView() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Card(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(
                  Icons.error_outline_rounded,
                  color: Colors.redAccent,
                  size: 48,
                ),
                const SizedBox(height: 14),
                Text(
                  'Erreur de chargement',
                  style: Theme.of(context).textTheme.titleLarge,
                ),
                const SizedBox(height: 10),
                Text(
                  _errorMessage ?? 'Une erreur est survenue.',
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
                const SizedBox(height: 18),
                ElevatedButton.icon(
                  onPressed: _loadPortfolio,
                  icon: const Icon(Icons.refresh_rounded),
                  label: const Text('Réessayer'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildPortfolioView() {
    return RefreshIndicator(
      onRefresh: _loadPortfolio,
      color: AppTheme.primaryColor,
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(20, 18, 20, 100),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildPortfolioHero(),
            const SizedBox(height: 20),
            _buildAnalyticsRow(),
            const SizedBox(height: 20),
            _buildChartCard(),
            const SizedBox(height: 20),
            Text(
              'Répartition des actifs',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 12),
            ..._assets.map(_buildAssetCard),
          ],
        ),
      ),
    );
  }

  Widget _buildPortfolioHero() {
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
            Icons.account_balance_wallet_rounded,
            color: Colors.white,
            size: 34,
          ),
          const SizedBox(height: 16),
          Text(
            'Total Portfolio Value',
            style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                  color: Colors.white.withOpacity(0.90),
                ),
          ),
          const SizedBox(height: 8),
          Text(
            '\$${_totalValue.toStringAsFixed(0)}',
            style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                  color: Colors.white,
                  fontSize: 32,
                ),
          ),
          const SizedBox(height: 14),
          Row(
            children: [
              _buildHeroBadge('Risque : $_riskLevel'),
              const SizedBox(width: 10),
              _buildHeroBadge(
                'Rendement estimé : ${_estimatedReturn.toStringAsFixed(1)}%',
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildHeroBadge(String text) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.16),
        borderRadius: BorderRadius.circular(30),
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

  Widget _buildAnalyticsRow() {
    return Row(
      children: [
        Expanded(
          child: _buildMiniStatCard(
            title: 'Diversification',
            value: '${_diversificationScore.toStringAsFixed(1)}/10',
            icon: Icons.hub_rounded,
            color: const Color(0xFF10B981),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: _buildMiniStatCard(
            title: 'Actifs',
            value: '${_assets.length}',
            icon: Icons.pie_chart_rounded,
            color: const Color(0xFFF59E0B),
          ),
        ),
      ],
    );
  }

  Widget _buildMiniStatCard({
    required String title,
    required String value,
    required IconData icon,
    required Color color,
  }) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            CircleAvatar(
              backgroundColor: color.withOpacity(0.12),
              child: Icon(icon, color: color),
            ),
            const SizedBox(height: 14),
            Text(
              title,
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            const SizedBox(height: 6),
            Text(
              value,
              style: Theme.of(context).textTheme.titleLarge,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildChartCard() {
    final colors = [
      const Color(0xFF6C4DFF),
      const Color(0xFF10B981),
      const Color(0xFFF59E0B),
      const Color(0xFF3B82F6),
      const Color(0xFFEF4444),
    ];

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Allocation du portefeuille',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 20),
            SizedBox(
              height: 240,
              child: Row(
                children: [
                  Expanded(
                    flex: 5,
                    child: PieChart(
                      PieChartData(
                        sectionsSpace: 3,
                        centerSpaceRadius: 42,
                        sections: List.generate(_assets.length, (index) {
                          final asset = _assets[index];
                          final percentage =
                              (asset['percentage'] as num).toDouble();

                          return PieChartSectionData(
                            color: colors[index % colors.length],
                            value: percentage,
                            title: '${percentage.toInt()}%',
                            radius: 62,
                            titleStyle: const TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.bold,
                              color: Colors.white,
                            ),
                          );
                        }),
                      ),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    flex: 4,
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: List.generate(_assets.length, (index) {
                        final asset = _assets[index];
                        return Padding(
                          padding: const EdgeInsets.symmetric(vertical: 6),
                          child: Row(
                            children: [
                              Container(
                                width: 12,
                                height: 12,
                                decoration: BoxDecoration(
                                  color: colors[index % colors.length],
                                  borderRadius: BorderRadius.circular(4),
                                ),
                              ),
                              const SizedBox(width: 8),
                              Expanded(
                                child: Text(
                                  asset['name'].toString(),
                                  style: Theme.of(context).textTheme.bodyMedium,
                                ),
                              ),
                            ],
                          ),
                        );
                      }),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAssetCard(Map<String, dynamic> asset) {
    final name = asset['name']?.toString() ?? 'Actif';
    final percentage = asset['percentage']?.toString() ?? '0';

    IconData icon = Icons.pie_chart_outline_rounded;
    Color iconColor = AppTheme.primaryColor;

    if (name.toLowerCase().contains('actions')) {
      icon = Icons.trending_up_rounded;
      iconColor = const Color(0xFF10B981);
    } else if (name.toLowerCase().contains('etf')) {
      icon = Icons.stacked_line_chart_rounded;
      iconColor = const Color(0xFF3B82F6);
    } else if (name.toLowerCase().contains('crypto')) {
      icon = Icons.currency_bitcoin_rounded;
      iconColor = const Color(0xFFF59E0B);
    } else if (name.toLowerCase().contains('cash')) {
      icon = Icons.savings_rounded;
      iconColor = const Color(0xFF6C4DFF);
    } else if (name.toLowerCase().contains('or')) {
      icon = Icons.workspace_premium_rounded;
      iconColor = const Color(0xFFEF4444);
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 18),
        child: Row(
          children: [
            CircleAvatar(
              radius: 24,
              backgroundColor: iconColor.withOpacity(0.12),
              child: Icon(icon, color: iconColor),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    name,
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Allocation stratégique',
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                ],
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: const Color(0xFFEDE7FA),
                borderRadius: BorderRadius.circular(30),
              ),
              child: Text(
                '$percentage%',
                style: const TextStyle(
                  color: AppTheme.primaryColor,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
