import 'package:flutter/material.dart';
import '../services/api_service.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {

  final ApiService _apiService = ApiService();

  Map<String, dynamic>? _profile;

  bool _isLoading = true;

  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _loadProfile();
  }

  Future<void> _loadProfile() async {

    try {

      final profile = await _apiService.getUserProfile(1);

      if (!mounted) return;

      setState(() {
        _profile = profile;
        _isLoading = false;
      });

    } catch (e) {

      if (!mounted) return;

      setState(() {
        _errorMessage = e.toString();
        _isLoading = false;
      });

    }
  }

  @override
  Widget build(BuildContext context) {

    return Scaffold(

      appBar: AppBar(
        title: const Text("Investor Profile"),
      ),

      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _errorMessage != null
              ? Center(child: Text(_errorMessage!))
              : buildProfile(),

    );
  }

  Widget buildProfile() {

    final name = _profile?["name"] ?? "Unknown";
    final risk = _profile?["risk_level"] ?? "Unknown";
    final horizon = _profile?["horizon"] ?? "Unknown";
    final objective = _profile?["objective"] ?? "Unknown";

    return Padding(
      padding: const EdgeInsets.all(20),

      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,

        children: [

          Card(
            child: ListTile(
              leading: const Icon(Icons.person),
              title: Text(name),
              subtitle: const Text("Investor Name"),
            ),
          ),

          const SizedBox(height: 10),

          Card(
            child: ListTile(
              leading: const Icon(Icons.warning),
              title: Text(risk),
              subtitle: const Text("Risk Level"),
            ),
          ),

          const SizedBox(height: 10),

          Card(
            child: ListTile(
              leading: const Icon(Icons.timeline),
              title: Text(horizon),
              subtitle: const Text("Investment Horizon"),
            ),
          ),

          const SizedBox(height: 10),

          Card(
            child: ListTile(
              leading: const Icon(Icons.flag),
              title: Text(objective),
              subtitle: const Text("Financial Objective"),
            ),
          ),
        ],
      ),
    );
  }
}


