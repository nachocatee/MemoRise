package com.tjjhtjh.memorise.domain.user.service;

import com.tjjhtjh.memorise.domain.user.exception.UserEmailDuplicateException;
import com.tjjhtjh.memorise.domain.user.repository.UserRepository;
import com.tjjhtjh.memorise.domain.user.repository.entity.Role;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import com.tjjhtjh.memorise.domain.user.service.dto.request.JoinRequest;
import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Service
@Transactional
@RequiredArgsConstructor
@Slf4j
public class UserServiceImpl implements UserService {

    private final UserRepository userRepository;

    @Override
    public void join(JoinRequest joinRequest) {
         if (userRepository.findByEmailAndIsDeletedFalse(joinRequest.getEmail()).isPresent()) {
             throw new UserEmailDuplicateException("이미 존재하는 이메일입니다.");
         }
         User user = new User(joinRequest.getEmail(), joinRequest.getNickname(), joinRequest.getProfile(), Role.MEMBER);
         userRepository.save(user);
    }

}
