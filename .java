import java.util.ArrayList;

// Course class to represent a university course
class Course {
    private String courseCode;
    private String courseName;
    private int credits;
    private int maxCapacity;
    private ArrayList<Student> enrolledStudents;
    
    public Course(String courseCode, String courseName, int credits, int maxCapacity) {
        this.courseCode = courseCode;
        this.courseName = courseName;
        this.credits = credits;
        this.maxCapacity = maxCapacity;
        this.enrolledStudents = new ArrayList<>();
    }
    
    public String getCourseCode() {
        return courseCode;
    }
    
    public String getCourseName() {
        return courseName;
    }
    
    public int getCredits() {
        return credits;
    }
    
    public int getAvailableSeats() {
        return maxCapacity - enrolledStudents.size();
    }
    
    public boolean isFull() {
        return enrolledStudents.size() >= maxCapacity;
    }
    
    public boolean enrollStudent(Student student) {
        if (isFull()) {
            System.out.println("Cannot enroll. Course " + courseCode + " is full.");
            return false;
        }
        
        if (enrolledStudents.contains(student)) {
            System.out.println("Student is already enrolled in " + courseCode);
            return false;
        }
        
        enrolledStudents.add(student);
        return true;
    }
    
    public boolean dropStudent(Student student) {
        return enrolledStudents.remove(student);
    }
    
    public void displayCourseInfo() {
        System.out.println("\n--- Course Information ---");
        System.out.println("Code: " + courseCode);
        System.out.println("Name: " + courseName);
        System.out.println("Credits: " + credits);
        System.out.println("Enrolled: " + enrolledStudents.size() + "/" + maxCapacity);
        System.out.println("Available Seats: " + getAvailableSeats());
    }
    
    public void listEnrolledStudents() {
        System.out.println("\nStudents enrolled in " + courseCode + ":");
        if (enrolledStudents.isEmpty()) {
            System.out.println("No students enrolled yet.");
        } else {
            for (Student student : enrolledStudents) {
                System.out.println("- " + student.getName() + " (ID: " + student.getStudentId() + ")");
            }
        }
    }
}

// Student class to represent a university student
class Student {
    private String studentId;
    private String name;
    private String major;
    private ArrayList<Course> registeredCourses;
    
    public Student(String studentId, String name, String major) {
        this.studentId = studentId;
        this.name = name;
        this.major = major;
        this.registeredCourses = new ArrayList<>();
    }
    
    public String getStudentId() {
        return studentId;
    }
    
    public String getName() {
        return name;
    }
    
    public String getMajor() {
        return major;
    }
    
    public boolean registerForCourse(Course course) {
        if (registeredCourses.contains(course)) {
            System.out.println(name + " is already registered for " + course.getCourseCode());
            return false;
        }
        
        if (course.enrollStudent(this)) {
            registeredCourses.add(course);
            System.out.println(name + " successfully registered for " + course.getCourseCode());
            return true;
        }
        return false;
    }
    
    public boolean dropCourse(Course course) {
        if (registeredCourses.remove(course)) {
            course.dropStudent(this);
            System.out.println(name + " dropped " + course.getCourseCode());
            return true;
        } else {
            System.out.println(name + " is not registered for " + course.getCourseCode());
            return false;
        }
    }
    
    public int getTotalCredits() {
        int total = 0;
        for (Course course : registeredCourses) {
            total += course.getCredits();
        }
        return total;
    }
    
    public void displayStudentInfo() {
        System.out.println("\n--- Student Information ---");
        System.out.println("ID: " + studentId);
        System.out.println("Name: " + name);
        System.out.println("Major: " + major);
        System.out.println("Total Credits: " + getTotalCredits());
    }
    
    public void listRegisteredCourses() {
        System.out.println("\nCourses registered by " + name + ":");
        if (registeredCourses.isEmpty()) {
            System.out.println("No courses registered yet.");
        } else {
            for (Course course : registeredCourses) {
                System.out.println("- " + course.getCourseCode() + ": " + course.getCourseName() + 
                                   " (" + course.getCredits() + " credits)");
            }
        }
    }
}

// University class to manage students and courses
class University {
    private String universityName;
    private ArrayList<Student> students;
    private ArrayList<Course> courses;
    
    public University(String universityName) {
        this.universityName = universityName;
        this.students = new ArrayList<>();
        this.courses = new ArrayList<>();
    }
    
    public void addStudent(Student student) {
        students.add(student);
        System.out.println("Student " + student.getName() + " added to " + universityName);
    }
    
    public void addCourse(Course course) {
        courses.add(course);
        System.out.println("Course " + course.getCourseCode() + " added to " + universityName);
    }
    
    public Student findStudent(String studentId) {
        for (Student student : students) {
            if (student.getStudentId().equals(studentId)) {
                return student;
            }
        }
        return null;
    }
    
    public Course findCourse(String courseCode) {
        for (Course course : courses) {
            if (course.getCourseCode().equals(courseCode)) {
                return course;
            }
        }
        return null;
    }
    
    public void listAllStudents() {
        System.out.println("\n=== All Students at " + universityName + " ===");
        if (students.isEmpty()) {
            System.out.println("No students enrolled.");
        } else {
            for (Student student : students) {
                System.out.println("- " + student.getName() + " (ID: " + student.getStudentId() + 
                                   ", Major: " + student.getMajor() + ")");
            }
        }
    }
    
    public void listAllCourses() {
        System.out.println("\n=== All Courses at " + universityName + " ===");
        if (courses.isEmpty()) {
            System.out.println("No courses available.");
        } else {
            for (Course course : courses) {
                System.out.println("- " + course.getCourseCode() + ": " + course.getCourseName() + 
                                   " (" + course.getCredits() + " credits, " + 
                                   course.getAvailableSeats() + " seats available)");
            }
        }
    }
}

// Main class to demonstrate the system
public class UniversitySystem {
    public static void main(String[] args) {
        // Create university
        University university = new University("Tech University");
        
        System.out.println("=== University Course Registration System ===\n");
        
        // Create courses
        Course cs101 = new Course("CS101", "Introduction to Programming", 3, 30);
        Course cs102 = new Course("CS102", "Data Structures", 4, 25);
        Course math201 = new Course("MATH201", "Calculus I", 4, 40);
        Course eng101 = new Course("ENG101", "English Composition", 3, 35);
        
        // Add courses to university
        university.addCourse(cs101);
        university.addCourse(cs102);
        university.addCourse(math201);
        university.addCourse(eng101);
        
        System.out.println();
        
        // Create students
        Student alice = new Student("S001", "Alice Johnson", "Computer Science");
        Student bob = new Student("S002", "Bob Smith", "Mathematics");
        Student charlie = new Student("S003", "Charlie Brown", "Computer Science");
        
        // Add students to university
        university.addStudent(alice);
        university.addStudent(bob);
        university.addStudent(charlie);
        
        System.out.println("\n" + "=".repeat(50));
        
        // Students register for courses
        System.out.println("\n--- Course Registration ---");
        alice.registerForCourse(cs101);
        alice.registerForCourse(math201);
        alice.registerForCourse(eng101);
        
        bob.registerForCourse(math201);
        bob.registerForCourse(cs101);
        
        charlie.registerForCourse(cs102);
        charlie.registerForCourse(cs101);
        
        System.out.println("\n" + "=".repeat(50));
        
        // Display student information
        alice.displayStudentInfo();
        alice.listRegisteredCourses();
        
        bob.displayStudentInfo();
        bob.listRegisteredCourses();
        
        System.out.println("\n" + "=".repeat(50));
        
        // Display course information
        cs101.displayCourseInfo();
        cs101.listEnrolledStudents();
        
        math201.displayCourseInfo();
        math201.listEnrolledStudents();
        
        System.out.println("\n" + "=".repeat(50));
        
        // Drop a course
        System.out.println("\n--- Dropping a Course ---");
        alice.dropCourse(eng101);
        
        alice.displayStudentInfo();
        alice.listRegisteredCourses();
        
        System.out.println("\n" + "=".repeat(50));
        
        // List all students and courses
        university.listAllStudents();
        university.listAllCourses();
        
        System.out.println("\n" + "=".repeat(50));
        
        // Demonstrate course capacity
        System.out.println("\n--- Testing Course Capacity ---");
        Course smallCourse = new Course("CS999", "Advanced Topics", 3, 2);
        university.addCourse(smallCourse);
        
        alice.registerForCourse(smallCourse);
        bob.registerForCourse(smallCourse);
        charlie.registerForCourse(smallCourse); // Should fail - course is full
        
        smallCourse.displayCourseInfo();
    }
}
