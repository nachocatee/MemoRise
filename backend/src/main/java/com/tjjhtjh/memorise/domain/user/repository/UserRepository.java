package com.tjjhtjh.memorise.domain.user.repository;

import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface UserRepository extends JpaRepository<User, Long>, UserSupportRepository {
    Optional<User> findByUserSeqAndIsDeletedFalse(Long userSeq);
    Optional<User> findByEmailAndIsDeletedFalse(String email);
    Optional<User> findByEmail(String email);

}
