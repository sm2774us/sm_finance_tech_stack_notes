package de.rieckpil.blog;

import com.codeborne.selenide.Condition;
import com.codeborne.selenide.Configuration;
import com.codeborne.selenide.junit5.ScreenShooterExtension;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.openqa.selenium.By;
import org.openqa.selenium.chrome.ChromeOptions;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.web.server.LocalServerPort;

import static com.codeborne.selenide.Selenide.*;
import static org.springframework.boot.test.context.SpringBootTest.WebEnvironment;

@Disabled
@SpringBootTest(webEnvironment = WebEnvironment.RANDOM_PORT)
public class BookStoreWT {

  @LocalServerPort
  private Integer port;

  @RegisterExtension
  public static ScreenShooterExtension extension =
    new ScreenShooterExtension().to("target/selenide");

  @BeforeAll
  public static void configureChromeDriver() {
    ChromeOptions chromeOptions = new ChromeOptions();
    chromeOptions.addArguments("--no-sandbox", "--disable-dev-shm-usage");

    Configuration.browserCapabilities = chromeOptions;
  }

  @Test
  public void shouldDisplayBooks() {

    Configuration.reportsFolder = "target/selenide";

    open("http://localhost:" + port + "/book-store");

    $(By.id("all-books")).shouldNot(Condition.exist);

    screenshot("pre_book_fetch");

    $(By.id("fetch-books")).click();
    $(By.id("all-books")).shouldBe(Condition.visible);
    $$(By.tagName("h1")).shouldHaveSize(1);
  }
}
