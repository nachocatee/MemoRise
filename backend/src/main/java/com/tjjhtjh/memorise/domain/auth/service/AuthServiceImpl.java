package com.tjjhtjh.memorise.domain.auth.service;

import com.tjjhtjh.memorise.domain.auth.service.dto.request.LoginRequest;
import com.tjjhtjh.memorise.domain.auth.service.dto.response.LoginResponse;
import com.tjjhtjh.memorise.domain.user.repository.UserRepository;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@Transactional(readOnly = true)
@RequiredArgsConstructor
@Slf4j
public class AuthServiceImpl implements AuthService {

    private final UserRepository userRepository;

    @Override
    public LoginResponse login(LoginRequest loginRequest) {
        if(userRepository.findByEmailAndIsDeletedFalse(loginRequest.getEmail()).isEmpty()) {
            return new LoginResponse(false, null);
        }
        User user = userRepository.findByEmailAndIsDeletedFalse(loginRequest.getEmail()).orElseThrow();
        return new LoginResponse(true, user.getUserSeq());
    }
}
