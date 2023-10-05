package com.tjjhtjh.memorise.global.config;

//import com.tjjhtjh.memorise.domain.auth.repository.HttpCookieOAuth2AuthorizationRequestRepository;
//import com.tjjhtjh.memorise.domain.auth.service.OAuth2AuthenticationFailureHandler;
//import com.tjjhtjh.memorise.domain.auth.service.OAuth2AuthenticationSuccessHandler;
//import com.tjjhtjh.memorise.domain.auth.service.CustomOAuth2UserService;
import com.tjjhtjh.memorise.global.jwt.controller.JwtAuthenticationFilter;
import com.tjjhtjh.memorise.global.jwt.service.JwtTokenProvider;
import lombok.AllArgsConstructor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.*;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;

import static org.springframework.security.config.Customizer.withDefaults;

@Configuration
@EnableWebSecurity
@AllArgsConstructor
public class SecurityConfig {

    private final JwtTokenProvider jwtTokenProvider;
//    private final CustomOAuth2UserService customOAuth2UserService;
//    private final OAuth2AuthenticationSuccessHandler oAuth2AuthenticationSuccessHandler;
//    private final OAuth2AuthenticationFailureHandler oAuth2AuthenticationFailureHandler;

//    public SecurityConfig(JwtTokenProvider jwtTokenProvider, CustomOAuth2UserService customOAuth2UserService, OAuth2AuthenticationSuccessHandler oAuth2AuthenticationSuccessHandler, OAuth2AuthenticationFailureHandler oAuth2AuthenticationFailureHandler) {
//        this.jwtTokenProvider = jwtTokenProvider;
//        this.customOAuth2UserService = customOAuth2UserService;
//        this.oAuth2AuthenticationSuccessHandler = oAuth2AuthenticationSuccessHandler;
//        this.oAuth2AuthenticationFailureHandler = oAuth2AuthenticationFailureHandler;
//    }

//    @Bean
//    public HttpCookieOAuth2AuthorizationRequestRepository cookieOAuth2AuthorizationRequestRepository() {
//        return new HttpCookieOAuth2AuthorizationRequestRepository();
//    }

    @Bean
    public BCryptPasswordEncoder encoder() {
        return new BCryptPasswordEncoder();
    }

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http.cors(withDefaults());
        http.csrf(CsrfConfigurer::disable);
        http.sessionManagement((session) -> session
                .sessionCreationPolicy(SessionCreationPolicy.STATELESS));
        http.formLogin(FormLoginConfigurer::disable);
        http.httpBasic(HttpBasicConfigurer::disable);
        http.headers(headers -> headers
                    .frameOptions(HeadersConfigurer.FrameOptionsConfig::sameOrigin
                    )
            );
        http.logout(LogoutConfigurer::disable);
        http.authorizeHttpRequests(authorize -> authorize
                        .requestMatchers("/**").permitAll()
                        .requestMatchers("/auth/**").permitAll()
                        .requestMatchers("/user/**").permitAll()
//                        .requestMatchers("/user/**").permitAll()
                        .anyRequest().authenticated()
            );
//        http.oauth2Login(oAuth2LoginConfigurer -> oAuth2LoginConfigurer
//                        .authorizationEndpoint(authorizationEndpointConfig -> authorizationEndpointConfig
//                                        .baseUri("/oauth2/authorization")
//                                        .authorizationRequestRepository(cookieOAuth2AuthorizationRequestRepository()))
//                        .redirectionEndpoint(redirectionEndpointConfig -> redirectionEndpointConfig
//                                        .baseUri("/*/oauth2/code/*"))
//                        .userInfoEndpoint(userInfoEndpointConfig -> userInfoEndpointConfig
//                                        .userService(customOAuth2UserService)
//                        )
//                        .successHandler(oAuth2AuthenticationSuccessHandler)
//                        .failureHandler(oAuth2AuthenticationFailureHandler)
//            );

//                .oauth2Login()
//                .authorizationEndpoint().baseUri("/oauth2/authorize")
//                .authorizationRequestRepository(cookieOAuth2AuthorizationRequestRepository())
//                .and()
//                .redirectionEndpoint()
//                .baseUri("/*/oauth2/code/*")
//                .and()
//                .userInfoEndpoint().userService(customOAuth2UserService)
//                .and()
//                .successHandler(oAuth2AuthenticationSuccessHandler)
//                .failureHandler(oAuth2AuthenticationFailureHandler)
//                .and()
        http.addFilterBefore(new JwtAuthenticationFilter(jwtTokenProvider), UsernamePasswordAuthenticationFilter.class);
        return http.build();
    }
}
