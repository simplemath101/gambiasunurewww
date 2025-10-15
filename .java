/**
 * HelloArgs.java
 * This program demonstrates command-line argument handling.
 * It prints a personalized greeting if a name is provided,
 * or a default greeting if no argument is given.
 */
public class HelloArgs {
    public static void main(String[] args) {
        // Check if any command-line arguments were provided
        // args.length gives us the number of arguments passed
        if (args.length > 0) {
            // If at least one argument exists, use the first one (args[0])
            // Print a personalized greeting with the provided name
            System.out.println("Hello, " + args[0] + "!");
        } else {
            // If no arguments were provided, print the default greeting
            System.out.println("Hello, World!");
        }
    }
}
