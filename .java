import java.util.Scanner;
import java.util.Random;

// Enum to represent different network connection states
enum NetworkState {
    CONNECTED("Connected", "Your network connection is stable. You can browse and stream content."),
    CONNECTING("Connecting", "Please wait while establishing connection. This may take a few moments."),
    DISCONNECTED("Disconnected", "No network connection detected. Check your WiFi or Ethernet cable."),
    WEAK_SIGNAL("Weak Signal", "Connection is unstable. Move closer to your router or switch to a different network."),
    LIMITED_CONNECTIVITY("Limited Connectivity", "Connected but no internet access. Restart your router or contact your ISP."),
    AIRPLANE_MODE("Airplane Mode", "All wireless connections are disabled. Turn off airplane mode to connect."),
    RECONNECTING("Reconnecting", "Attempting to restore connection. Please wait..."),
    ERROR("Error", "A network error occurred. Try restarting your device or check network settings.");
    
    // Fields for each enum constant
    private final String displayName;
    private final String advice;
    
    // Constructor
    NetworkState(String displayName, String advice) {
        this.displayName = displayName;
        this.advice = advice;
    }
    
    // Getters
    public String getDisplayName() {
        return displayName;
    }
    
    public String getAdvice() {
        return advice;
    }
    
    // Method to display state information
    public void displayStateInfo() {
        System.out.println("\n╔════════════════════════════════════════════════════════════╗");
        System.out.println("  Network Status: " + displayName);
        System.out.println("╠════════════════════════════════════════════════════════════╣");
        System.out.println("  Advice: " + advice);
        System.out.println("╚════════════════════════════════════════════════════════════╝");
    }
    
    // Method to get icon representation
    public String getIcon() {
        switch (this) {
            case CONNECTED:
                return "✓";
            case CONNECTING:
            case RECONNECTING:
                return "↻";
            case DISCONNECTED:
                return "✗";
            case WEAK_SIGNAL:
                return "⚠";
            case LIMITED_CONNECTIVITY:
                return "!";
            case AIRPLANE_MODE:
                return "✈";
            case ERROR:
                return "⊗";
            default:
                return "?";
        }
    }
}

// Class to simulate a network connection
class NetworkConnection {
    private NetworkState currentState;
    private String connectionName;
    
    public NetworkConnection(String connectionName) {
        this.connectionName = connectionName;
        this.currentState = NetworkState.DISCONNECTED;
    }
    
    public NetworkState getCurrentState() {
        return currentState;
    }
    
    public void setState(NetworkState newState) {
        System.out.println("\n→ Changing state from " + currentState.getDisplayName() + 
                          " to " + newState.getDisplayName() + "...");
        this.currentState = newState;
    }
    
    public void displayStatus() {
        System.out.println("\n═══════════════════════════════════════════════════════════════");
        System.out.println("  Connection: " + connectionName);
        System.out.println("  Status: [" + currentState.getIcon() + "] " + 
                          currentState.getDisplayName());
        System.out.println("═══════════════════════════════════════════════════════════════");
    }
    
    public void diagnose() {
        displayStatus();
        currentState.displayStateInfo();
        provideTroubleshootingSteps();
    }
    
    private void provideTroubleshootingSteps() {
        System.out.println("\nTroubleshooting Steps:");
        
        switch (currentState) {
            case CONNECTED:
                System.out.println("  • No action needed - connection is working properly");
                System.out.println("  • Run a speed test if experiencing slowness");
                break;
                
            case CONNECTING:
            case RECONNECTING:
                System.out.println("  • Wait 30-60 seconds for connection to establish");
                System.out.println("  • If it takes too long, restart your device");
                break;
                
            case DISCONNECTED:
                System.out.println("  • Check if WiFi is turned on");
                System.out.println("  • Verify cable connections");
                System.out.println("  • Restart your router and modem");
                break;
                
            case WEAK_SIGNAL:
                System.out.println("  • Move closer to the wireless router");
                System.out.println("  • Remove obstacles between device and router");
                System.out.println("  • Consider using a WiFi extender");
                break;
                
            case LIMITED_CONNECTIVITY:
                System.out.println("  • Restart your router by unplugging for 30 seconds");
                System.out.println("  • Check if other devices can connect");
                System.out.println("  • Contact your Internet Service Provider");
                break;
                
            case AIRPLANE_MODE:
                System.out.println("  • Open Settings");
                System.out.println("  • Turn off Airplane Mode");
                System.out.println("  • Wait for network to reconnect");
                break;
                
            case ERROR:
                System.out.println("  • Run network diagnostics tool");
                System.out.println("  • Reset network settings");
                System.out.println("  • Update network drivers");
                break;
        }
    }
}

// Main class
public class NetworkStateManager {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        NetworkConnection connection = new NetworkConnection("WiFi Network");
        
        System.out.println("╔════════════════════════════════════════════════════════════╗");
        System.out.println("║       NETWORK CONNECTION STATE MANAGER                     ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝");
        
        boolean running = true;
        
        while (running) {
            System.out.println("\n\n--- MENU ---");
            System.out.println("1. Check Current Status");
            System.out.println("2. Change Network State");
            System.out.println("3. Run Diagnostics");
            System.out.println("4. Simulate Random State Change");
            System.out.println("5. Show All Possible States");
            System.out.println("6. Exit");
            System.out.print("\nEnter your choice: ");
            
            try {
                int choice = scanner.nextInt();
                scanner.nextLine(); // Consume newline
                
                switch (choice) {
                    case 1:
                        connection.displayStatus();
                        System.out.println("\nAdvice: " + connection.getCurrentState().getAdvice());
                        break;
                        
                    case 2:
                        displayStateOptions();
                        System.out.print("\nSelect state (1-8): ");
                        int stateChoice = scanner.nextInt();
                        scanner.nextLine();
                        
                        NetworkState[] states = NetworkState.values();
                        if (stateChoice >= 1 && stateChoice <= states.length) {
                            connection.setState(states[stateChoice - 1]);
                            connection.getCurrentState().displayStateInfo();
                        } else {
                            System.out.println("Invalid choice!");
                        }
                        break;
                        
                    case 3:
                        connection.diagnose();
                        break;
                        
                    case 4:
                        simulateRandomState(connection);
                        break;
                        
                    case 5:
                        showAllStates();
                        break;
                        
                    case 6:
                        running = false;
                        System.out.println("\nExiting Network State Manager. Goodbye!");
                        break;
                        
                    default:
                        System.out.println("\nInvalid choice! Please try again.");
                }
            } catch (Exception e) {
                System.out.println("\nInvalid input! Please enter a number.");
                scanner.nextLine(); // Clear invalid input
            }
        }
        
        scanner.close();
    }
    
    private static void displayStateOptions() {
        System.out.println("\n--- Available Network States ---");
        NetworkState[] states = NetworkState.values();
        for (int i = 0; i < states.length; i++) {
            System.out.println((i + 1) + ". [" + states[i].getIcon() + "] " + 
                             states[i].getDisplayName());
        }
    }
    
    private static void simulateRandomState(NetworkConnection connection) {
        Random random = new Random();
        NetworkState[] states = NetworkState.values();
        NetworkState randomState = states[random.nextInt(states.length)];
        
        System.out.println("\n🎲 Simulating random network event...");
        connection.setState(randomState);
        connection.diagnose();
    }
    
    private static void showAllStates() {
        System.out.println("\n╔════════════════════════════════════════════════════════════╗");
        System.out.println("║           ALL POSSIBLE NETWORK STATES                      ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝");
        
        for (NetworkState state : NetworkState.values()) {
            System.out.println("\n[" + state.getIcon() + "] " + state.getDisplayName());
            System.out.println("    → " + state.getAdvice());
        }
    }
}
